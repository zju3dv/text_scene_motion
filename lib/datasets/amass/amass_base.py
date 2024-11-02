from pathlib import Path
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from lib.utils import logger
from lib.datasets.make_dataset import DATASET


@DATASET.register()
class AMASS(data.Dataset):
    def __init__(self, dat_cfg, split='train'):
        super().__init__()
        self.split = split
        self.max_motion_len = dat_cfg.get('max_motion_len', 120)
        self.num_scene_points = dat_cfg.get('num_scene_points', 1024)
        self.interval = dat_cfg.get('interval', 60)
        self.sample_data_interval = dat_cfg.get('sample_data_interval', 1)
        
        # file path
        self.amass_root = Path(dat_cfg.amass_root)
        
        # load data path
        self.datapaths = self._load_datapaths()
        self._datapaths2meta()
        
        self.preload = dat_cfg.get('preload', True)
        if self.preload:
            self._preload_npz_datas()
            
        self._load_scene()
        self.idx2meta = self.idx2meta[::self.sample_data_interval]
        
     
    def _load_datapaths(self):
        test_splits = ['TotalCapture']
        # test_splits = []
        data_paths = []
        for dir in self.amass_root.iterdir():
            if self.split == 'train' and dir.name not in test_splits:
                data_paths += dir.glob('*/*.npz')
            elif self.split == 'test' and dir.name in test_splits:
                data_paths += dir.glob('*/*.npz')
        return data_paths
        
    
    def _datapaths2meta(self):
        self.idx2meta = []
        for datapath in self.datapaths:
            nframes = int(datapath.stem.split('_')[-4])
            for start in range(0, max(0, nframes - self.max_motion_len) + 1, self.interval):
                end = min(start + self.max_motion_len, nframes)
                if end - start < 20:
                    continue
                meta = {
                    'start': start,
                    'end': end,
                    'datapath': str(datapath),
                }
                self.idx2meta.append(meta)
    
    def _preload_npz_datas(self):
        self.npz_datas = {}
        for datapath in tqdm(self.datapaths):
            npz_data = np.load(datapath, allow_pickle=True)
            npz_dict = {
                    'joints': npz_data['joints'],
                    'trans': npz_data['trans'],
                    'root_orient': npz_data['root_orient'],
                    'betas': npz_data['betas'],
                    'pose_body': npz_data['pose_body'],
                    'floor_height': npz_data['floor_height'],
            }
            self.npz_datas[str(datapath)] = npz_dict


    def _load_scene(self):
        self.floor_normal = np.zeros((self.num_scene_points, 3))
        self.floor_normal[:, 2] = 1.0
    
    
    def __getitem__(self, index):
        meta = self.idx2meta[index]
        data = {}
        meta.update({
            'idx': index,
            'split': self.split,
            'dataname': 'amass',
        })
        smplx_params, motion_mask, joints, floor_height = self.get_motion(meta)
        smplx_params = {k: torch.FloatTensor(v) for k, v in smplx_params.items()}
        xyz_az = self.get_scene(joints, floor_height)
        
        data.update({
            'meta': meta,
            'smplx_params_az': smplx_params,
            'motion_mask': torch.BoolTensor(motion_mask),
            'xyz_az': torch.FloatTensor(xyz_az),
            'normal': torch.FloatTensor(self.floor_normal),
            'betas': smplx_params['betas'][0],
        })
        return data
        
    
    def __len__(self):
        return len(self.idx2meta)
    
    
    def get_motion(self, meta):
        if self.preload:
            npz_data = self.npz_datas[meta['datapath']]
        else:
            npz_data = np.load(meta['datapath'], allow_pickle=True)
        st, ed = meta['start'], meta['end']
        joints = npz_data['joints'][st:ed]
        
        smplx_params = {
            'transl': npz_data['trans'][st:ed],
            'global_orient': npz_data['root_orient'][st:ed],
            'betas': npz_data['betas'],
            'body_pose': npz_data['pose_body'][st:ed],
        }
        # pad motion
        S = len(joints)
        meta.update({
            'm_len': S,
        })
        smplx_params['betas'] = np.tile(smplx_params['betas'], (S, 1))
        if S == self.max_motion_len:
            pass
        elif S > self.max_motion_len:
            for k, d in smplx_params.items():
                smplx_params[k] = d[: self.max_motion_len]
            joints = joints[: self.max_motion_len]
        else:
            for k, d in smplx_params.items():
                padding = np.tile(d[-1], (self.max_motion_len - S, 1))
                smplx_params[k] = np.concatenate([d, padding])
            padding = np.tile(joints[-1], (self.max_motion_len - S, 1, 1))
            joints = np.concatenate([joints, padding])
        motion_mask = np.zeros(self.max_motion_len, dtype=bool)
        motion_mask[S:] = 1
        
        return smplx_params, motion_mask, joints, npz_data['floor_height']
    
    
    def get_scene(self, joints, floor_height):
        joints = joints.reshape(-1, 3)
        joints_mean = joints.mean(axis=0)
        radius = np.linalg.norm(joints - joints_mean, axis=1).max()
        floor = (np.random.rand(self.num_scene_points, 3) * 2.0 - 1.0) * radius + joints_mean
        floor[:, 2] = - floor_height
        return floor