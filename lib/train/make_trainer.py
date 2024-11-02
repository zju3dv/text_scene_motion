from lib.utils import logger
from lib.wrapper.wrapper import WRAPPER
from .trainer import Trainer

def _wrapper_factory(cfg, network):
    wrapper_cfg = cfg.wrapper_cfg
    logger.info(f'Making network wrapper @ {wrapper_cfg.name}')
    network_wrapper = WRAPPER.get(wrapper_cfg.name)(network, cfg, wrapper_cfg)
    return network_wrapper

def make_trainer(cfg, network):
    logger.info(f"Making trainer @ {cfg.trainer.name}")
    network = _wrapper_factory(cfg, network)
    return Trainer(network, cfg)
