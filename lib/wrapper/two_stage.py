from .preprocess import *
from .postprocess import *
from .visualization import *


def sample_two_stage_midprocess(wrapper, batch):
    batch['sample_stage1_transl'] = batch['ksamples_transl']
    batch[f'sample_stage1_orient_6d'] = batch['ksamples_orient_6d']
    batch[f'sample_stage1_orient'] = batch[f'ksamples_orient']
    return batch['ksamples']
    
    
def save_sample(cfg, batch, traj=False):
    dataname = batch['meta'][0]['dataname']
    coord = 'hm' if dataname == 'humanise' else 'prox'
    return_samples = []

    # gt
    gt_smplx = batch[f'smplx_params_{cfg.coord}']
    gt_t_to_scannet = batch['t_to_scannet']
    if 'gt' in batch: # use pred_center, t_to_scannet
        if 't_to_scannet' in batch['gt']:
            gt_t_to_scannet = batch['gt']['t_to_scannet']
    gt_tgt_center = batch[f'obj_center_{coord}'] if 'gt' not in batch else batch['gt'][f'obj_center_{coord}']
    gt_tgt_mask = batch[f'obj_mask'] if 'gt' not in batch else batch['gt'][f'obj_mask']
    
    for b in range(len(batch['meta'])):
        if cfg.action != 'all' and cfg.action != batch['meta'][b]['action']:
            continue
        output_dict = {}
        nframes = batch['meta'][b]['m_len']
        
        output_dict = {
            'meta': batch['meta'][b],
            'pred_params': {
                'betas': gt_smplx['betas'][b, 0].cpu(),
                'trans': batch['ksamples_transl'][b, :, :nframes].cpu(),
                'orient': batch['ksamples_orient'][b, :, :nframes].cpu(),
                'tgt_center': batch[f'obj_center_{coord}'][b].cpu(),
                't_to_scannet': batch['t_to_scannet'][b].cpu(),
                'tgt_mask': batch['obj_mask'][b].cpu(),
            },
            'scene_id': batch['meta'][b]['scene_id'],
            'anchor_index': batch['meta'][b]['anchor'],
        }

        if dataname == 'humanise':
            output_dict.update({
                'gt_params': {
                    'betas': gt_smplx['betas'][b, 0].cpu(),
                    'trans': gt_smplx['transl'][b, :nframes].cpu(),
                    'orient': gt_smplx['global_orient'][b, :nframes].cpu(),
                    'pose_body': gt_smplx['body_pose'][b, :nframes].cpu(),
                    'tgt_center': gt_tgt_center[b].cpu(),
                    't_to_scannet': gt_t_to_scannet[b].cpu(),
                    'tgt_mask': gt_tgt_mask[b].cpu(),
                },
            })   
        
        if not traj:
            output_dict['pred_params'].update({
                'pose_body': batch['ksamples_body_pose'][b, :, :nframes].cpu(),  # [K, T, 63]
            })
        return_samples.append(output_dict)
    return return_samples
    