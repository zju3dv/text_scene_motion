import numpy as np
import torch
from lib.utils import logger
from lib.utils.comm import all_gather
from lib.utils.smplx_utils import load_smpl_faces
from .metrics import METRIC


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.local_rank = cfg.local_rank
        self.metric_func_names = cfg.metrics
        self.body_faces = torch.from_numpy(load_smpl_faces()).cuda()
        self.coord = cfg.net_cfg.coord
        
        logger.info(f"Metrics Functions: {self.metric_func_names}")
        self.init_metric_stats()

    def init_metric_stats(self):
        """ Call at initialization and end """
        self.metric_stats = {}

    def update(self, k, v_list: list):
        """ v_list need to be List of simple scalars """
        if k in self.metric_stats:
            self.metric_stats[k].extend(v_list)
        else:
            self.metric_stats[k] = v_list

    def evaluate(self, batch):
        for k in self.metric_func_names:
            METRIC.get(f'{k}_metric')(self, batch)

    def summarize(self):
        if len(self.metric_stats) == 0:
            return {}, {}

        values = [np.array(all_gather(self.metric_stats[k])).flatten() for k in self.metric_stats]
        metrics_raw = {k: vs for k, vs in zip(self.metric_stats, values)}
        metrics = {k: np.mean(vs) for k, vs in zip(self.metric_stats, values)}

        message = f"Avg-over {len(values[0])}. Metrics: "
        for k, v in metrics.items():
            message += f'{k}: {v:.4f} ; '
        if self.local_rank == 0:
            logger.info(message)

        self.init_metric_stats()
        return metrics, metrics_raw


def make_evaluator(cfg):
    return Evaluator(cfg)
