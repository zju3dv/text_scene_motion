import torch
import torch.nn as nn
import torch.distributions as dist
from einops import rearrange
from .make_model import MODEL
from .utils import MLPBlock, ResTimeMLPBlock, PositionalEncoding
from .sensor import SENSOR

class MotionFromCondCFG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_x = cfg.d_x
        self.coord = cfg.coord
        self.d_model = cfg.get('d_model', 512)
        self.arch = cfg.get('arch', 'trans_dec')
        self._build_time_emb(cfg)
        self._build_in_layer(cfg)
        self._build_out_layer(cfg)
        self._build_cond_layer(cfg)
        self._build_transformer_layer(cfg)
    
    def _build_time_emb(self, cfg):
        self.sequence_pos_encoder = PositionalEncoding(self.d_model, dropout=0.0)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
    def _build_in_layer(self, cfg):
        self.in_layer = MLPBlock(self.d_x, self.d_model)
        
    def _build_out_layer(self, cfg):
        self.res_layer = ResTimeMLPBlock(
            self.d_model * 2,
            self.d_model,
            self.d_model,
        )
        self.out_layer = MLPBlock(self.d_model, self.d_x)
        
    def _build_cond_layer(self, cfg):
        self.d_betas = cfg.get('d_betas', 10)
        self.beta_encoder = MLPBlock(self.d_betas, self.d_model)
    
    def _build_transformer_layer(self, cfg):
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu")
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=4)
        elif self.arch == 'trans_dec':
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu")
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=4)
    
    def get_bs_device(self, batch):
        return batch['motion_mask'].shape[0], batch['motion_mask'].device
    
    def get_beta_feat(self, batch):
        if 'beta_feat' not in batch:
            beta = batch['betas'][:, :self.d_betas]
            beta_feat = self.beta_encoder(beta[:, None])
            beta_feat = rearrange(beta_feat, 'b l c -> l b c')
            batch.update({
                'beta_feat': beta_feat,
            })
        return batch['beta_feat']
    
    def text_condition(self, batch):
        if 'text_feature' not in batch:
            bs, device = self.get_bs_device(batch)
            text_feat = torch.zeros((bs, self.d_model), device=device)
            batch.update({
                'text_feature': text_feat,
            })
        text_feat = batch['text_feature']
        return text_feat[None]
    
    def condition(self, batch, x_t):
        return None, None
    
    def forward(self, x_t, ts, batch):
        t_emb = self.sequence_pos_encoder.query(ts)
        t_emb = self.time_mlp(t_emb)
        t_emb = rearrange(t_emb, 'b l c -> l b c')
        
        x = self.in_layer(x_t)
        res = x.clone()
        x = rearrange(x, 'b l c -> l b c')
        
        cond, cond_mask = self.condition(batch, x_t)
        extra_num = cond.shape[0]
        mask = torch.cat((cond_mask, batch['motion_mask']), dim=-1)
        xseq = torch.cat((cond, x), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        
        if self.arch == 'trans_enc':
            output = self.seqTransEncoder(tgt=xseq, memory=t_emb, tgt_key_padding_mask=mask)
        elif self.arch == 'trans_dec':
            output = self.seqTransDecoder(tgt=xseq, memory=t_emb, tgt_key_padding_mask=mask)
        output = rearrange(output, 'l b c -> b l c')
        output = output[:, extra_num:]
        
        pred = torch.cat([res, output], dim=-1)
        pred = self.res_layer(pred, rearrange(t_emb, 'l b c -> b l c'))
        pred = self.out_layer(output)
        return pred
    
    
@MODEL.register()
class MotionFromSceneTextTrajVoxelV0(MotionFromCondCFG):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def _build_cond_layer(self, cfg):
        super()._build_cond_layer(cfg)
        self.build_sensors(cfg)
        
    def _build_sensor(self, sensor_cfg):
        sensor_cfg.coord = self.coord
        return SENSOR.get(sensor_cfg.name)(sensor_cfg)
    
    def build_sensors(self, cfg):
        self.env_sensor = self._build_sensor(cfg.env_sensor)
        self.target_sensor = self._build_sensor(cfg.target_sensor)
        self.traj_sensor = self._build_sensor(cfg.traj_sensor)
        
        self.env_sensor_fuse_layer = nn.Sequential(
            nn.Linear(7*(self.env_sensor.voxel_dim**3), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
        )
        
        self.target_sensor_fuse_layer = nn.Sequential(
            nn.Linear(7*(self.target_sensor.voxel_dim**3), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
        )
        
        self.traj_sensor_fuse_layer = nn.Sequential(
            nn.Linear(9+7*(self.traj_sensor.voxel_dim**3), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
        )
        
    def get_sensor_feat(self, batch, x_t):
        bs, device = self.get_bs_device(batch)
        if 'env_sensors' not in batch:
            env_voxel, env_points, env_nn_normal = self.env_sensor.forward(batch)
            env_sensors = torch.cat((env_voxel, env_points, env_nn_normal), dim=-1).reshape(bs, 1, -1)
            batch['env_sensors'] = self.env_sensor_fuse_layer(env_sensors)
        
        if 'target_sensors' not in batch:
            target_voxel, target_points, target_nn_normal = self.target_sensor.forward(batch)
            target_sensors = torch.cat((target_voxel, target_points, target_nn_normal), dim=-1).reshape(bs, 1, -1)
            batch['target_sensors'] = self.target_sensor_fuse_layer(target_sensors)
        
        # should update every loop
        x_unnorm = batch['normalizer'].unnormalize(x_t)
        traj_voxels, traj_points, traj_nn_normals = self.traj_sensor.forward(batch, x_unnorm)
        traj_sensors = torch.cat((traj_voxels, traj_points, traj_nn_normals), dim=-1).reshape(bs, x_unnorm.shape[1], -1)
        traj_sensors = torch.cat((batch['traj'], traj_sensors), dim=-1)
        traj_sensors = self.traj_sensor_fuse_layer(traj_sensors)

        sensors = torch.cat((batch['env_sensors'], batch['target_sensors'], traj_sensors), dim=1)
        sensors = rearrange(sensors, 'b l c -> l b c')
        # mask
        if 'motion_sensor_mask' not in batch:
            cond_mask = torch.zeros((bs, 2), device=device) # (b, nc)
            traj_mask = batch['motion_mask']
            batch['motion_sensor_mask'] = torch.cat((cond_mask, traj_mask), dim=-1)
        return sensors, batch['motion_sensor_mask']
        
        
    def condition(self, batch, x_t):
        bs, device = self.get_bs_device(batch)
        beta_feat = self.get_beta_feat(batch)
        text_feat = self.text_condition(batch)
        cond_mask = torch.zeros((bs, 2), device=device) # (b, nc)
        sensor_feat, sensor_mask = self.get_sensor_feat(batch, x_t)
        feats = torch.cat((beta_feat, text_feat, sensor_feat), dim=0)
        cond_mask = torch.cat((cond_mask, sensor_mask), dim=-1)
        return feats, cond_mask
        
    
@MODEL.register()
class TrajFromSceneTextVoxelV0(MotionFromSceneTextTrajVoxelV0):
    def build_sensors(self, cfg):
        self.env_sensor = self._build_sensor(cfg.env_sensor)
        self.target_sensor = self._build_sensor(cfg.target_sensor)
        
        self.env_sensor_fuse_layer = nn.Sequential(
            nn.Linear(7*(self.env_sensor.voxel_dim**3), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
        )
        
        self.target_sensor_fuse_layer = nn.Sequential(
            nn.Linear(7*(self.target_sensor.voxel_dim**3), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
        )
        
    def get_sensor_feat(self, batch, x_t):
        bs, device = self.get_bs_device(batch)
        if 'env_sensors' not in batch:
            env_voxel, env_points, env_nn_normal = self.env_sensor.forward(batch)
            env_sensors = torch.cat((env_voxel, env_points, env_nn_normal), dim=-1).reshape(bs, 1, -1)
            batch['env_sensors'] = self.env_sensor_fuse_layer(env_sensors)
        
        if 'target_sensors' not in batch:
            target_voxel, target_points, target_nn_normal = self.target_sensor.forward(batch)
            target_sensors = torch.cat((target_voxel, target_points, target_nn_normal), dim=-1).reshape(bs, 1, -1)
            batch['target_sensors'] = self.target_sensor_fuse_layer(target_sensors)
        
        sensors = torch.cat((batch['env_sensors'], batch['target_sensors']), dim=1)
        sensors = rearrange(sensors, 'b l c -> l b c')
        # mask
        if 'traj_sensor_mask' not in batch:
            batch['traj_sensor_mask'] = torch.zeros((bs, 2), device=device) # (b, nc)
        return sensors, batch['traj_sensor_mask']