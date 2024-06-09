import torch.nn as nn
from lib.utils import logger
from lib.utils.registry import Registry
MODEL = Registry('model')
from .eps_model import *

def make_model(cfg):
    logger.info(f'Making Model @ {cfg.name}')
    return MODEL.get(cfg.name)(cfg)