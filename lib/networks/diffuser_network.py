import torch
import torch.nn as nn
from .make_network import NETWORK
from .diffusion.make_diffuser import make_diffuser
from lib.utils.geo_transform import axis_angle_to_rot_6d, rot_6d_to_axis_angle

@NETWORK.register()
class DiffuserNetwork(nn.Module):
    def __init__(self, net_cfg):
        super().__init__()
        self._set_cfg(net_cfg)
        net_cfg.model.coord = self.coord
        self.diffuser = make_diffuser(net_cfg)
    
    def _set_cfg(self, cfg):
        self.k_sample = cfg.get('k_sample', 1)
        self.repr = cfg.repr # motion or traj
        self.coord = cfg.coord
        self.d_l = cfg.model.d_l
        self.d_x = cfg.model.d_x
    
    def pre_gt(self, batch):
        smplx_params = batch[f'smplx_params_{self.coord}']
        batch['orient_6d'] = orient_rot6d = axis_angle_to_rot_6d(smplx_params['global_orient'])
        batch['body_pose_6d'] = pose_rot6d = axis_angle_to_rot_6d(smplx_params['body_pose'])
        x_unnorm = torch.cat((smplx_params['transl'], orient_rot6d, pose_rot6d), dim=-1)
        x_norm = batch['normalizer'].normalize(x_unnorm)
        # start pose
        batch['start_pose'] = x_norm[:, [0], :]
        if self.repr == 'motion':
            if 'stage1_traj' in batch:
                batch['traj'] = batch['stage1_traj']
            else:
                batch['traj'] = x_norm[..., :9]
        elif self.repr == 'traj':
            # first 9 is traj
            x_norm = x_norm[..., :9]
        return x_norm
    
    def pre_inference(self, batch):
        smplx_params = batch.get('start_params', None)
        if smplx_params is not None:
            orient_rot6d = axis_angle_to_rot_6d(smplx_params['global_orient'])
            pose_rot6d = axis_angle_to_rot_6d(smplx_params['body_pose'])
            x_unnorm = torch.cat((smplx_params['transl'], orient_rot6d, pose_rot6d), dim=-1)
            x_norm = batch['normalizer'].normalize(x_unnorm)
            # start pose
            batch['start_pose'] = x_norm
        if self.repr == 'motion':
            if 'stage1_traj' in batch:
                batch['traj'] = batch['stage1_traj']
        
    def post(self, batch, output, key):
        unnorm = batch['normalizer'].unnormalize(output)
        batch[f'{key}_transl'] = unnorm[..., :3]
        batch[f'{key}_orient_6d'] = orient_6d = unnorm[..., 3:9]
        batch[f'{key}_orient'] = rot_6d_to_axis_angle(orient_6d)
        if self.repr == 'motion':
            batch[f'{key}_body_pose_6d'] = body_pose_6d = unnorm[..., 9:]
            batch[f'{key}_body_pose'] = rot_6d_to_axis_angle(body_pose_6d)
    
    def forward(self, batch):
        # preprocess
        batch['x'] = self.pre_gt(batch)
        gt, recon = self.diffuser(batch)
        # batch['gt'] = gt
        # postprocess
        self.post(batch, recon, 'recon')
    
    @torch.no_grad()
    def inference(self, batch):
        # this x is only used to provide shape information
        batch['x'] = torch.zeros((len(batch['meta']), self.d_l, self.d_x)).to(batch['motion_mask'].device)
        init_x = batch.get('init_x', None)
        if init_x is not None and self.repr == 'traj':
            init_x = init_x[..., :9]
        # TODO: add inference
        self.pre_inference(batch)
        k = self.k_sample
        batch['ksamples'] = ksamples = self.diffuser.sample(batch, k=k, init_x=init_x)
        # postprocess
        self.post(batch, ksamples, 'ksamples')