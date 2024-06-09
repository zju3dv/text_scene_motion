from pathlib import Path
import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils import data
from pyquaternion import Quaternion as Q

from lib.utils import logger


action_anchor = {
    'sit': -1,
    'stand up': 0,
    'walk': -1,
    'lie': -1,
}

class Humanise(data.Dataset):
    def __init__(self, dat_cfg, split='train'):
        super().__init__()
        self.split = split
        self.action = dat_cfg.action
        self.coord = dat_cfg.coord
        logger.info(f'action: {self.action}')
        self.use_color = dat_cfg.get('use_color', True)
        self.use_normal = dat_cfg.get('use_normal', False)
        self.num_scene_points = dat_cfg.get('num_scene_points', 32768)
        self.num_object_points = dat_cfg.get('num_object_poitns', 512)
        self.max_motion_len = dat_cfg.get('max_motion_len', 120)

        # file path
        self.humanise_root = dat_cfg.humanise_root
        self.scannet_root = dat_cfg.scannet_root
        self.humanise_prep_root = dat_cfg.humanise_prep_root
        self.scannet_prep_root = dat_cfg.scannet_prep_root
        # load data
        pkl_file = os.path.join(self.humanise_prep_root, f'all_{self.split}_data.pkl')
        Path(self.humanise_prep_root).mkdir(exist_ok=True, parents=True)
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                pdata = pickle.load(f)
            self.idx2meta = pdata['idx2meta']
            self.id2motion = pdata['id2motion']
            self.id2scene = pdata['id2scene']
        else:
            self.id2motion = dict() # motion_id -> motion_data
            self.id2scene = dict() # scene_id -> scene_data
            self.idx2meta = self._split2meta() # idx -> meta
            save_dict = {
                'idx2meta': self.idx2meta,
                'id2motion': self.id2motion,
                'id2scene': self.id2scene,
            }
            with open(pkl_file, 'wb') as f:
                pickle.dump(save_dict, f)
        
        # action specific
        if self.action != 'all':
            if self.action == 'standup':
                self.idx2meta = [meta for meta in self.idx2meta if meta['action'] == 'stand up']
            else:
                self.idx2meta = [meta for meta in self.idx2meta if meta['action'] == self.action]
        # sorted
        sorted_idx2meta = sorted(self.idx2meta, key=lambda x: x['anno_path'])
        for i, meta in enumerate(sorted_idx2meta):
            meta.update({
                'unique_idx': i,
            })
       
    def __len__(self):
        return len(self.idx2meta)
    
    def _split2meta(self):
        split2meta = list()
        for action in ['lie', 'sit', 'stand up', 'walk']:
            splitaction2meta = list()
            all_pkls = (glob.glob(os.path.join(self.humanise_root, f'align_data_release/{action}/*/anno.pkl')))
            for pkl in tqdm(all_pkls):
                all_meta = self.read_anno_meta(str(pkl))
                for data_id, meta in enumerate(all_meta):
                    meta.update({
                        'anno_path': str(pkl),
                        'data_id': data_id,
                    })
                    if self.split == 'train' and int(meta['scene_id'][5:9]) < 600:
                        self._update_id2motion_data(meta)
                        self._update_id2scene_data(meta)
                        splitaction2meta.append(meta)
                        meta.update({
                            'm_len': self.id2motion[meta['motion_id']]['m_len'],
                        })
                    elif self.split == 'test' and int(meta['scene_id'][5:9]) >= 600:
                        self._update_id2motion_data(meta)
                        self._update_id2scene_data(meta)
                        splitaction2meta.append(meta)
                        meta.update({
                            'm_len': self.id2motion[meta['motion_id']]['m_len'],
                        })
                            
            split2meta.extend(splitaction2meta)
            logger.debug(f'{self.split}: {action}: {len(splitaction2meta)}')
        
        return split2meta
    
    def read_anno_meta(self, anno_path: str, data_id: int=-1):
        with open(anno_path, 'rb') as fp:
            all_data = pickle.load(fp)
            all_data = [all_data[data_id]] if data_id != -1 else all_data
        all_info = list()
        for anno_data in all_data:
            meta = {
                # 1. text
                'action': anno_data['action'],
                'anchor': action_anchor[anno_data['action']],
                'utterance': anno_data['utterance'],
                # 2. scene
                'scene_id': anno_data['scene'],
                'scene_trans': np.array(anno_data['scene_translation']),
                'scene_ply_path': str(Path(self.scannet_root)/ 'scans' / anno_data['scene'] / f"{anno_data['scene']}_vh_clean_2.ply"),
                'scene_npy_path': os.path.join(self.scannet_prep_root, f"{anno_data['scene']}.npy"),
                # 3. motion
                'motion_id': anno_data['motion'],
                'motion_trans': anno_data['translation'],
                'motion_rotat': anno_data['rotation'],
                'motion_path': str(Path(self.humanise_root) / 'pure_motion' / anno_data['action'] / anno_data['motion'] / 'motion.pkl'),
                # 4. object
                'object_id': anno_data['object_id'],
                'object_name': anno_data['object_label'],
                # 'object_semantic_label': anno_data['object_semantic_label'],
            }
            all_info.append(meta)
        return all_info
    
    def _update_id2motion_data(self, meta):
        if meta['motion_id'] not in self.id2motion:
            self.id2motion.update({
                meta['motion_id']: self.read_pure_motion(meta),
            })
    
    def _update_id2scene_data(self, meta):
        if meta['scene_id'] not in self.id2scene:
            self.id2scene.update({
                meta['scene_id']: self.read_scene_data(meta),
            })
    
    def read_pure_motion(self, meta: dict):
        with open(meta['motion_path'], 'rb') as f:
            motion_data = pickle.load(f)
        gender, org_trans, org_orient, betas, pose_body, pose_hand, _, _, joints = motion_data
        # meta.update({
        #     'gender': str(gender),
        # })
        smplx_params = {
            'betas': betas[:10],
            'transl': org_trans,
            'global_orient': org_orient,
            'body_pose': pose_body,
            'left_hand_pose': pose_hand[:, :45],
            'right_hand_pose': pose_hand[:, 45:],
        }
        motion_dict = {
            'smplx_params': smplx_params,
            'pelvis': joints[:, 0, :], # (N, 3)
            'm_len': len(joints),
        }
        return motion_dict
    
    def get_motion(self, meta: dict):
        motion_id = meta['motion_id']
        motion_dict = self.id2motion[motion_id]
        smplx_params = {k: v.copy() for k,v in motion_dict['smplx_params'].items()}
        pelvis = motion_dict['pelvis'].copy()
        
        # get T_pure2hm
        T1 = np.eye(4)
        T1[:2, -1] = - pelvis[meta['anchor'], :2]
        T2 = Q(axis=[0, 0, 1], angle=meta['motion_rotat']).transformation_matrix
        T2[0:3, -1] = meta['motion_trans']
        T_pure2hm = T2 @ T1
        # pad motion
        S = len(pelvis)
        meta.update({
            'm_len': S,
        })
        smplx_params['betas'] = np.tile(smplx_params['betas'], (S, 1))
        if S > self.max_motion_len:
            for k, d in smplx_params.items():
                smplx_params[k] = d[: self.max_motion_len]
            pelvis = pelvis[: self.max_motion_len]
            meta.update({
                'm_len': self.max_motion_len,
            })
        else:
            for k, d in smplx_params.items():
                padding = np.tile(d[-1], (self.max_motion_len - S, 1))
                smplx_params[k] = np.concatenate([d, padding])
            padding = np.tile(pelvis[-1], (self.max_motion_len - S, 1))
            pelvis = np.concatenate([pelvis, padding])
        motion_mask = np.zeros(self.max_motion_len, dtype=bool)
        motion_mask[S:] = 1
        
        return smplx_params, motion_mask, pelvis, T_pure2hm
        
    
    def read_scene_data(self, meta: dict):
        scene_prep_data = np.load(meta['scene_npy_path'])
        sel_idx = np.random.randint(0, len(scene_prep_data), (self.num_scene_points))
        scene_data = scene_prep_data[sel_idx]
        scene_dict = {
            'xyz_scan': scene_data[:, :3],
            'color': scene_data[:, 3:6],
            'normal': scene_data[:, 6:9],
            'obj_label': scene_data[:, -1],
        }
        return scene_dict

    def get_scene_object(self, meta):
        scene_id = meta['scene_id']
        scene_dict = self.id2scene[scene_id]
        # scene
        xyz_scan = scene_dict['xyz_scan']
        xyz_hm = xyz_scan + meta['scene_trans']
        color = scene_dict['color']
        normal = scene_dict['normal']
        # object
        obj_id = meta['object_id']
        obj_label = scene_dict['obj_label']
        obj_mask = obj_label == obj_id # target object mask
        obj_center_hm = np.mean(xyz_hm[obj_mask], axis=0)
        return xyz_hm, color, normal, obj_mask, obj_center_hm, obj_label