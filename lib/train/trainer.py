import time
import datetime
import torch
from torch.nn.parallel import DistributedDataParallel
import tqdm

from lib.utils import logger
from lib.utils.net_utils import to_cuda


class Trainer(object):
    def __init__(self, network, cfg):
        self.cfg = cfg.clone()
        network.cuda(cfg.local_rank)
        if cfg.distributed:
            network = DistributedDataParallel(
                network, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=True)
        self.network = network
        self.local_rank = cfg.local_rank

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def train(self, epoch, data_loader, optimizer, recorder):
        if self.local_rank == 0:
            logger.info(f"Training: Epoch {epoch}, {self.cfg.task}")
        self.network.train()
        end = time.time()

        if self.cfg.distributed:
            data_loader.sampler.set_epoch(epoch)

        max_iter = len(data_loader)
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch)
            batch['cur_epoch'] = epoch
            batch = self.network(batch)

            # training stage: loss; optimizer; scheduler
            optimizer.zero_grad()
            loss = batch["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 0.5)
            optimizer.step()

            if self.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1
            loss_stats = self.reduce_loss_stats(batch["loss_stats"])
            recorder.update_loss_stats(loss_stats)
            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % self.cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

            if iteration % self.cfg.rec_interval == 0 or iteration == (max_iter - 1):
                recorder.record('train')

    @torch.no_grad()
    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        if self.cfg.local_rank == 0:
            logger.info(f"Validation / Testing: Epoch {epoch}")
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)

        for batch in tqdm.tqdm(data_loader):
            batch = to_cuda(batch)
            batch = self.network(batch)
            
            loss_stats = self.reduce_loss_stats(batch["loss_stats"])
            if evaluator is not None:
                evaluator.evaluate(batch)

            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))

        if evaluator is not None:
            result, result_raw = evaluator.summarize()
            if recorder:
                recorder.record('val_metric', epoch, result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats)

