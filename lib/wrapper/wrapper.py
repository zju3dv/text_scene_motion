import torch.nn as nn
import numpy as np
import torch
import pickle
import clip

from lib.utils import logger
from lib.utils.smplx_utils import make_smplx, load_smpl_faces
from lib.utils.normalize import make_normalizer
from lib.utils.registry import Registry
WRAPPER = Registry('wrapper')
from .loss import LOSS
from .supervision import SUP
from .preprocess import PRE
from .postprocess import POST
from .two_stage import *


@WRAPPER.register()
class Wrapper(nn.Module):
    def __init__(self, net, cfg,
                 wrapper_cfg):
        super().__init__()
        # config
        self.cfg = cfg
        self.vis3d = None
        self.loss_weights = {k: v for k, v in cfg.loss_weights.items() if v > 0}
        self.pre_methods = wrapper_cfg.pre_methods
        self.post_methods = wrapper_cfg.post_methods
        self.sup_methods = wrapper_cfg.sup_methods
        self.vis_methods = wrapper_cfg.vis_methods
        logger.info(f'Preprocess method: {self.pre_methods}')
        logger.info(f'Postprocess method: {self.post_methods}')
        logger.info(f'Supervision method: {self.sup_methods}')
        logger.info(f'Visualization method: {self.vis_methods}')

        # main network
        self.net = net
        self.inference = False
        
        # smplx model
        self.smplx_model = make_smplx(wrapper_cfg.smplx_model_type)
        self.smplx_faces = torch.from_numpy(load_smpl_faces())
        
        self._build_text_model(wrapper_cfg)
    
    def _build_text_model(self, cfg):
        # clip
        self.clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu',
                                                jit=False)  # Must set jit=False for training
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
    
    def forward(self, batch, inference=False, compute_supervision=True, compute_loss=True):
        self.inference = inference
        self.preprocess(batch)
        if inference:
            self.net.inference(batch)
        else:
            self.net(batch)
        self.postprocess(batch)
        
        if compute_supervision:
            self.compute_supervision(batch)
        if compute_loss:
            self.compute_loss(batch)

        return batch  # important for DDP

    def preprocess(self, batch):
        for name in self.pre_methods:
            PRE.get(f'process_{name}')(self, batch)
            
    def postprocess(self, batch):
        for name in self.post_methods:
            POST.get(f'get_{name}')(self, batch)

    def compute_supervision(self, batch):
        for name in self.sup_methods:
            SUP.get(f'get_{name}')(self, batch)

    def compute_loss(self, batch):
        B = len(batch['meta'])
        loss = 0.
        loss_stats = {}
        for k, v in self.loss_weights.items():
            cur_loss = v * LOSS.get(f'cal_{k}_loss')(self, batch, loss_stats)
            assert cur_loss.shape[0] == B
            loss += cur_loss
        loss_stats.update({'loss_weighted_sum': loss.detach().cpu()})
        batch.update({"loss": loss, "loss_stats": loss_stats})


@WRAPPER.register()
class MotionDiffuserWrapper(Wrapper):
    def __init__(self, net, cfg, wrapper_cfg) -> None:
        super().__init__(net, cfg, wrapper_cfg)
        self.coord = cfg.net_cfg.coord
        self.normalizer = None
        if wrapper_cfg.get('normalizer', None) is not None:
            self.normalizer = self._build_normalizer(wrapper_cfg.normalizer)
    
    def _build_normalizer(self, cfg):
        with open(cfg.file, 'rb') as f:
            fdata = pickle.load(f)
            xmin, xmax = fdata['xmin'], fdata['xmax']
        return make_normalizer(cfg.name, (xmin, xmax))
            
    def forward(self, batch, inference=False, compute_supervision=True, compute_loss=True):
        if self.normalizer is not None:
            batch['normalizer'] = self.normalizer
        return super().forward(batch, inference, compute_supervision, compute_loss)


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
