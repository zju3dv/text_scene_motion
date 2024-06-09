from termcolor import colored
from lib.utils import logger
from wis3d.wis3d import Wis3D
from pathlib import Path
import os
import time


vis3d_colors = {
    # Usage:
    # mesh.visual.vertex_colors = vis3d_colors['cyan']
    # vis3d.add_mesh(mesh, name='mesh-name')
    'cyan': [2, 252, 155, 255],
    'blue': [2, 163, 252, 255],
    'khaki': [252, 186, 2, 255],
    'orange': [252, 60, 2, 255],

    'white': [255, 222, 173, 255],
    'gray': [119, 136, 153, 255],

    'green_gray': [46, 139, 87, 255],
    'green': [2, 252, 14, 255],
    'red_gray': [205, 85, 85, 255],
    'red': [220, 0, 0, 255],
}


def make_vis3d(cfg, exp_name=None, record_dir='out/vis3d/', time_postfix=False):
    if cfg is not None and exp_name is not None:
        exp_name = f"{cfg.task}@{exp_name}"
    elif exp_name is None:
        exp_name = f"{cfg.task}"
    if time_postfix:
        exp_name = f'{exp_name}_{int(time.time()) % 10000:04d}'
    log_dir = Path(record_dir) / exp_name
    if log_dir.exists():
        logger.warning(colored(f'remove contents of directory {log_dir}', 'red'))
        os.system(f"rm -rf {log_dir}")
    log_dir.mkdir(parents=True)
    logger.info(f"Making directory: {log_dir}")
    vis3d = Wis3D(record_dir, exp_name)
    return vis3d
