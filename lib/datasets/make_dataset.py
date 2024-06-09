import os
import torch
from time import time
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from lib.utils import logger
from lib.utils.registry import Registry
DATASET = Registry('dataset')
from .humanise.humanise_motion import HumaniseMotion


def make_dataset(cfg, split='train'):
    tic = time()
    dat_cfg = cfg.get(f"{split}_dat")
    logger.info(f"Making {split} dataset: {dat_cfg.name}")

    # load real dataset
    dataset = DATASET.get(dat_cfg.name)(cfg.dat_cfg, dat_cfg.split)
    limit_size = dat_cfg.limit_size
    if limit_size > 0 and len(dataset) > limit_size:
        logger.warning(f"Working on subset of size {limit_size}")
        dataset = torch.utils.data.Subset(dataset, list(range(limit_size)))

    logger.debug(f"Time for making dataset: {time() - tic:.2f}s")
    return dataset

def collate_fn_wrapper(batch):
    keys_to_collate_as_list = ['meta']
    list_in_batch = {}
    for k in keys_to_collate_as_list:
        if k in batch[0]:
            list_in_batch[k] = [data[k] for data in batch]
    # use default collate for the rest of batch
    batch = default_collate(batch)
    batch.update({k: v for k, v in list_in_batch.items()})
    return batch


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def make_data_loader(cfg, split='train'):
    dataset = make_dataset(cfg, split)
    logger.info(f"Final {split} dataset size: {len(dataset)}")

    datloader_cfg = cfg.get(split)
    batch_size = datloader_cfg.batch_size
    num_workers = datloader_cfg.num_workers

    sampler = make_data_sampler(dataset, datloader_cfg.shuffle, cfg.distributed)

    # assume 1*node with N*Gpus: evenly adjust batchsize and num_workers
    if cfg.distributed:
        assert batch_size % int(os.environ['WORLD_SIZE']) == 0
        batch_size = batch_size // int(os.environ['WORLD_SIZE'])
        num_workers = num_workers // int(os.environ['WORLD_SIZE'])

    collate_fn = collate_fn_wrapper
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers= (split == 'train' and num_workers > 0),
        collate_fn=collate_fn,
    )

    return dataloader