import os
import torch
from .optimizers.radam import RAdam
from lib.utils import logger

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net, is_train, net_params=None):
    cfg_optimizer = cfg.train.optimizer if is_train else cfg.test.optimizer
    optim = cfg_optimizer.optim

    # set-up learning rate
    lr = cfg_optimizer.lr
    if lr == 0:
        world_bs = cfg.train.batch_size
        lr = cfg_optimizer.canonical_lr * (world_bs / cfg_optimizer.canonical_bs)
    logger.info(f"lr {lr:.2e}, world batchsize {world_bs}")

    adam_decay = cfg_optimizer.weight_decay
    adamw_decay = cfg_optimizer.adamw_weight_decay

    parameters = net_params if net_params else net.parameters()
    if optim == 'adam':
        optimizer = _optimizer_factory[optim](
            parameters, lr=lr, weight_decay=adam_decay)
    elif optim == 'adamw':
        optimizer = _optimizer_factory[optim](
            parameters, lr=lr, weight_decay=adamw_decay)
    else:
        raise ValueError

    return optimizer
