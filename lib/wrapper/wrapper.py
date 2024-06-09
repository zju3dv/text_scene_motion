import torch.nn as nn
import numpy as np
import torch
import pickle
import clip

from lib.utils.smplx_utils import make_smplx, load_smpl_faces
from lib.utils.normalize import make_normalizer
from lib.utils.registry import Registry
WRAPPER = Registry('wrapper')
from .preprocess import PRE
from .postprocess import POST
from .two_stage import *

@WRAPPER.register()
class TwoStageWrapper(nn.Module):
    def __init__(self, path_net, motion_net, cfg,
                 traj_cfg, motion_cfg):
        super().__init__()
        # config
        self.cfg = cfg
        self.coord = cfg.coord

        # main network
        self.path_net = path_net
        self.motion_net = motion_net
        self.inference = False
        
        self.pre_methods = traj_cfg.wrapper_cfg.pre_methods
        self.post_methods = motion_cfg.wrapper_cfg.post_methods
        
        # smplx model
        self.smplx_model = make_smplx('humanise')
        self.smplx_faces = torch.from_numpy(load_smpl_faces())
        self.rot_repr = '6d'
        if cfg.get('normalizer', None) is not None:
            self.normalizer = self._build_normalizer(cfg.normalizer)
        self._build_text_model()
            
    def _build_normalizer(self, cfg):
        with open(cfg.file, 'rb') as f:
            fdata = pickle.load(f)
            xmin, xmax = fdata['xmin'], fdata['xmax']
        return make_normalizer(cfg.name, (xmin, xmax))
    
    def _build_text_model(self):
        # clip
        self.clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu',
                                                jit=False)  # Must set jit=False for training
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
    
    def forward(self, batch, inference=False, compute_supervision=False, compute_loss=False):
        if self.normalizer is not None:
            batch['normalizer'] = self.normalizer
        self.inference = inference
        
        if inference:
            for name in self.pre_methods:
                PRE.get(f'process_{name}')(self, batch)
                
            self.path_net.inference(batch)
            sample_traj = sample_two_stage_midprocess(self, batch)
            sample_k_motion = []
            for k in range(sample_traj.shape[1]):
                batch['stage1_traj'] = sample_traj[:, k]
                self.motion_net.inference(batch)
                sample_k_motion.append(batch['ksamples'])
            batch['ksamples'] = torch.stack(sample_k_motion, dim=1)[:, :, 0]
            
            self.motion_net.post(batch, batch['ksamples'], 'ksamples')

            for name in self.post_methods:
                POST.get(f'get_{name}')(self, batch)

        return batch  # important for DDP

    def get_vis3d_name(self, batch, inference=False):
        meta = batch['meta'][0]
        split = meta['split']
        idx = str(meta['idx'])
        return f"{split}-{idx}"
