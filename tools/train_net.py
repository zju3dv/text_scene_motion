from lib.utils import offscreen_flag
import argparse
from lib.config import make_cfg, save_cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.evaluators import make_evaluator
from lib.train import make_trainer, make_recorder, make_lr_scheduler, set_lr_scheduler, make_optimizer
from lib.utils.comm import setup_distributed, clear_directory_for_training
from lib.utils.net_utils import save_network, load_network

import torch
torch.autograd.set_detect_anomaly(True)


def train(cfg):
    data_loader = make_data_loader(cfg, split='train')
    val_loader = make_data_loader(cfg, split='test')
    
    resume = cfg.resume
    if cfg.local_rank == 0:
        clear_directory_for_training(cfg.record_dir, resume)
        save_cfg(cfg, resume)

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)
    # optimizer
    optimizer = make_optimizer(cfg, network, is_train=True)
    # scheduler
    if cfg.train.get('scheduler', None) is not None:
        scheduler = make_lr_scheduler(cfg, optimizer)
        set_lr_scheduler(cfg, scheduler)

    epoch_start = load_network(network, resume, cfg, epoch=-1)
    for epoch in range(epoch_start+1, cfg.train.epoch+2):
        recorder.epoch = epoch
        trainer.train(epoch, data_loader, optimizer, recorder)
        if cfg.train.get('scheduler', None) is not None:
            scheduler.step()
        # save
        if epoch % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_network(network, cfg.model_dir, epoch)
        # eval
        if epoch % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='main')
    parser.add_argument('--cfg_file', '-c', type=str, required=True)
    parser.add_argument('--is_test', action='store_true', default=False)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)
    if cfg.distributed:
        setup_distributed()
    train(cfg)