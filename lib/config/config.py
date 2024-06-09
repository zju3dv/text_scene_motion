import os
from .yacs import CfgNode
from contextlib import redirect_stdout
from lib.utils import logger, logger_activate_debug_mode

# ===== Default Configs ===== #
_cfg = CfgNode()
_cfg.gpus = [0]
_cfg.task = ''
_cfg.is_train = True
_cfg.debug = False
_cfg.resume = False
_cfg.save_ep = 10
_cfg.eval_ep = 20
_cfg.model_dir = 'auto'
_cfg.record_dir = 'auto'
_cfg.log_interval = 20
_cfg.rec_interval = 20


def make_cfg(args):
    ''' args: requires 'cfg_file' key, should support `.` access '''
    logger.info(f'Making config: {args.cfg_file}')  
    # 0. merge all configs
    cfg = _cfg.clone()
    default_cfg_path = f"configs/{args.cfg_file.split('/')[1]}/default.yaml"
    if os.path.exists(default_cfg_path):
        cfg.merge_from_file(default_cfg_path)
        if 'dataset_cfg_path' in cfg.keys():
            cfg.merge_from_file(cfg.dataset_cfg_path)
        cfg.merge_from_file(args.cfg_file)
        # dirs
        if cfg.record_dir == 'auto':
            cfg.record_dir = f'out/train/{cfg.task}'
        os.system(f'mkdir -p {cfg.record_dir}')
        if cfg.model_dir == 'auto':
            cfg.model_dir = f'out/train/{cfg.task}/model'
        os.system(f'mkdir -p {cfg.model_dir}')
    else:  # loading from trained model
        cfg.merge_from_file(args.cfg_file)
        logger.warning('overwrite gpus and resume!')
        cfg.gpus = [0]
        cfg.resume = True
    cfg.merge_from_list(getattr(args, 'opts', []))
    cfg.is_train = not args.is_test

    # 1. Auto config devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in cfg.gpus])
    os.environ['EGL_DEVICE_ID'] = str(min(cfg.gpus))
    cfg.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    cfg.distributed = os.environ.get('LOCAL_RANK', None) != None
    
    # 2. Set logger
    if cfg.task == 'debug':
        logger_activate_debug_mode()
        logger.debug(cfg)
    return cfg


def save_cfg(cfg: CfgNode, resmue=False):
    filename = f'{cfg.record_dir}/config'
    if resmue:
        filename += '.resume'
    filename += '.yaml'
    with open(filename, 'w') as f:
        logger.info(f'Saving cfg into {filename}')
        with redirect_stdout(f):
            print(cfg)
