from lib.utils import logger
from lib.utils.registry import Registry
NETWORK = Registry('network')
from .diffuser_network import DiffuserNetwork

def make_network(cfg):
    name = cfg.net_cfg.name
    logger.info("Making network: {}".format(name))
    network = NETWORK.get(name)(net_cfg=cfg.net_cfg)
    return network
