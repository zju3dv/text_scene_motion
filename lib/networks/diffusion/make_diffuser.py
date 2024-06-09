import torch.nn as nn
from lib.networks.model.make_model import make_model
from lib.utils.registry import Registry
DIFFUSER = Registry('diffuser')
from .ddpm_base import *

def make_diffuser(cfg) -> nn.Module:
    ## use diffusion model, the model is a eps model
    eps_model = make_model(cfg.model)
    diffuser  = DIFFUSER.get(cfg.diffuser.name)(eps_model, cfg.diffuser)
    return diffuser