import numpy as np
from pathlib import Path
import torch
import numpy as np
import pickle
from pyquaternion import Quaternion as Q
from pytorch3d.transforms import axis_angle_to_matrix
from lib.utils.geo_transform import matrix_to_axis_angle, apply_T_on_points

action_anchor = {
    'sit': -1,
    'stand up': 0,
    'walk': -1,
    'lie': -1,
}


def transform_smplx(
        T: torch.FloatTensor, # (4, 4)
        origin_trans: torch.FloatTensor,
        origin_orient: torch.FloatTensor,
        origin_pelvis: torch.FloatTensor,
    ):
    pelvis = apply_T_on_points(origin_pelvis[None], T[None])[0]
    trans = pelvis + (origin_trans - origin_pelvis)
    orient = (T[:3, :3] @ axis_angle_to_matrix(origin_orient)).inverse()
    orient = matrix_to_axis_angle(orient)
    return trans, orient, pelvis

def transform_smplx_batch(
        T: torch.FloatTensor, # (B, 4, 4)
        origin_trans: torch.FloatTensor,
        origin_orient: torch.FloatTensor,
        origin_pelvis: torch.FloatTensor,
    ):
    pelvis = apply_T_on_points(origin_pelvis, T)
    trans = pelvis + (origin_trans - origin_pelvis)
    orient = (T[:, None, :3, :3] @ axis_angle_to_matrix(origin_orient)).transpose(-1, -2)
    orient = matrix_to_axis_angle(orient)[0]
    return trans, orient, pelvis


def transform_motion(meta: dict, motion_data):
    gender, org_trans, org_orient, betas, pose_body, pose_hand, _, _, joints = motion_data
    anchor_frame = action_anchor[meta['action']]
    meta.update({
        'gender': str(gender),
        'anchor': anchor_frame,
    })
    org_pelvis = joints[:, 0, :]
    T1 = np.eye(4)
    T1[:2, -1] = - org_pelvis[anchor_frame, :2]
    T2 = Q(axis=[0, 0, 1], angle=meta['motion_rotat']).transformation_matrix
    T3 = np.eye(4)
    T3[0:3, -1] = meta['motion_trans']
    trans_body = T3 @ T2 @ T1
    smplx_params = {
        'betas': np.tile(betas[:10], (len(org_trans), 1)),
        'transl': org_trans,
        'global_orient': org_orient,
        'body_pose': pose_body,
        'left_hand_pose': pose_hand[:, :45],
        'right_hand_pose': pose_hand[:, 45:],
    }
    return smplx_params, org_pelvis, trans_body


def read_anno_meta(anno_path: str, data_id: int=-1):
    with open(anno_path, 'rb') as fp:
        all_data = pickle.load(fp)
        all_data = [all_data[data_id]] if data_id != -1 else all_data
    all_info = list()
    for anno_data in all_data:
        meta = {
            'action': anno_data['action'],
            'utterance': anno_data['utterance'],
            'scene_id': anno_data['scene'],
            'scene_npy_path': str(Path('pkl_data/scannet/preprocess') / f"{anno_data['scene']}.npy"),
            'scene_ply_path': str(Path('datalinks/ScanNet/scans') / anno_data['scene'] / f"{anno_data['scene']}_vh_clean_2.ply"),
            'motion_id': anno_data['motion'],
            'motion_path': str(Path('datalinks/HUMANISE/pure_motion') / anno_data['action'] / anno_data['motion'] / 'motion.pkl'),
            'object_id': anno_data['object_id'],
            'object_name': anno_data['object_label'],
            'object_semantic_label': anno_data['object_semantic_label'],
            # matrix
            'motion_trans': anno_data['translation'],
            'motion_rotat': anno_data['rotation'],
            'scene_translation': np.array(anno_data['scene_translation']),
        }
        all_info.append(meta)
    return all_info


def read_pure_motion(meta: dict):
    with open(meta['motion_path'], 'rb') as f:
        motion_data = pickle.load(f)
    return motion_data


def read_anno(anno_path: str, data_id: int=-1):
    meta_data = read_anno_meta(anno_path, data_id)
    all_info = list()
    for meta in meta_data:
        motion_data = read_pure_motion(meta)
        smplx_params = transform_motion(meta, motion_data)
        scene_T = np.eye(4)
        scene_T[:3, -1] = meta['scene_translation']
        data_info = {
            'meta': meta,
            'smplx_params': smplx_params,
            'scene_T': scene_T,
        }
        all_info.append(data_info)
    return all_info
