import torch
from .humanise_base import Humanise
from lib.datasets.make_dataset import DATASET
from .utils import transform_smplx


@DATASET.register()
class HumaniseMotion(Humanise):
    def __init__(self, dat_cfg, split='train') -> None:
        super().__init__(dat_cfg, split=split)
        self.coord = dat_cfg.get('coord', 'oc')
        self.min_t = dat_cfg.get('min_t', 0)
        self.idx2meta = [meta for meta in self.idx2meta if meta['m_len'] >= self.min_t]
        self.sample_data_interval = dat_cfg.get('sample_data_interval', 1)
        
    def __getitem__(self, idx):
        meta = self.idx2meta[idx]
        meta.update({
            'split': self.split,
            'dataname': 'humanise',
            'idx': idx,
        })
        data = {
            'meta': meta,
        }
        self.process_scene_object_motion(data)
        return data
        
    def process_scene_object_motion(self, data):
        meta = data['meta']
        # 1. process scene and object
        xyz_hm, color, normal, obj_mask, obj_center_hm, obj_label = self.get_scene_object(meta)
        
        # 2. process motion
        smplx_params, motion_mask, pelvis_pure, T_pure2hm = self.get_motion(meta)
        smplx_params = {k: torch.FloatTensor(v) for k, v in smplx_params.items()}
        
        # 3. trans motion from pure to hm
        smplx_params_hm = {k: v.clone() for k, v in smplx_params.items()}
        smplx_params_hm['transl'], smplx_params_hm['global_orient'], pelvis_hm = transform_smplx(
            torch.FloatTensor(T_pure2hm),
            smplx_params['transl'],
            smplx_params['global_orient'],
            torch.FloatTensor(pelvis_pure))
        t_to_scannet = - meta['scene_trans']
        
        # 4. trans motion from hm to oc
        if self.coord == 'oc':
            t_hm2oc = - obj_center_hm
            t_to_scannet -= t_hm2oc
            # trans scene points
            xyz_oc = xyz_hm + t_hm2oc
            obj_center_oc = obj_center_hm + t_hm2oc
            # trans smplx params
            smplx_params_oc = {k: v.clone() for k, v in smplx_params_hm.items()}
            smplx_params_oc['transl'] += torch.FloatTensor(t_hm2oc)
        
        data.update({
            'betas': smplx_params['betas'][0],
            'motion_mask': torch.BoolTensor(motion_mask), # (M, 1)
            # scene
            'color': torch.FloatTensor(color),
            'normal': torch.FloatTensor(normal),
            'obj_mask': torch.BoolTensor(obj_mask),
            'obj_label': torch.LongTensor(obj_label),
            # hm
            't_to_scannet': torch.FloatTensor(t_to_scannet),
            't_sn2hm': torch.FloatTensor(meta['scene_trans']),
            'xyz_hm': torch.FloatTensor(xyz_hm),
            'obj_center_hm': torch.FloatTensor(obj_center_hm),
            'smplx_params_hm': smplx_params_hm,
        })
        
        if self.coord == 'oc':
            data.update({
                't_hm2oc': torch.FloatTensor(t_hm2oc),
                'xyz_oc': torch.FloatTensor(xyz_oc),
                'obj_center_oc': torch.FloatTensor(obj_center_oc),
                'smplx_params_oc': smplx_params_oc,
            })