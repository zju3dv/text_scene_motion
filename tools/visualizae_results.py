import os
import argparse
import pickle
import trimesh
import torch
import numpy as np

from lib.config import make_cfg
from lib.utils import logger
from lib.utils.vis3d_utils import make_vis3d
from lib.utils.smplx_utils import make_smplx

def main():
    all_motion_data = pickle.load(open(cfg.save_path, 'rb'))
    vis3d = make_vis3d(None, f'vis_output_release_{cfg.vis_id}', 'out/vis3d')
    sample = all_motion_data[cfg.vis_id]
    # print input text
    utterance = sample['meta']['utterance']
    logger.info(f'input text: {utterance}')
    # load scene
    scene_id = sample['scene_id']
    scene_path = os.path.join(cfg.scannet_root, f'scans/{scene_id}/{scene_id}_vh_clean_2.ply')
    scene_mesh = trimesh.load(scene_path)
    # load predicted motion
    params_ = sample['pred_params']
    K = params_['trans'].size(0)
    T = params_['trans'].size(1)
    smplx_params = {
        'betas': params_['betas'][None],
        'global_orient': params_['orient'].reshape(K*T, 3),  # [KT, 3]
        'transl':  params_['trans'].reshape(K*T, 3) + params_['t_to_scannet'],
        'body_pose':  params_['pose_body'].reshape(K*T, -1)}
    smplx_params = {k: v.to(device) for k, v in smplx_params.items()}
    smplx_output = smplx_model(**smplx_params)
    pred_k_motions = smplx_output.vertices.reshape(K, T, -1, 3) # (K, T, V, 3)
    # visualize
    for t in range(T):
        vis3d.set_scene_id(t)
        body_mesh = trimesh.Trimesh(vertices=pred_k_motions[cfg.k_id, t].detach().cpu().numpy(), faces=smplx_face)
        vis3d.add_mesh(body_mesh, name=f'pred_body')
        vis3d.add_mesh(scene_mesh, name='scene')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c", default='configs/test/generate.yaml')
    parser.add_argument("--is_test", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device)
    device = torch.device('cuda')
    smplx_model = make_smplx('humanise').to(device)
    smplx_face = torch.from_numpy(smplx_model.bm.faces.astype(np.int64))

    main()