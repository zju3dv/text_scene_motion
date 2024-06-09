import torch
from typing import Any
from lib.utils import logger
from lib.utils.registry import Registry
NORMALIZER = Registry("normalizer")


@NORMALIZER.register()
class NormalizerPoseMotion():
    def __init__(self, xmin_max: Any) -> None:
        self.xmin = xmin_max[0]
        self.xmax = xmin_max[1]

    def normalize(self, x: Any) -> Any:
        shape = x.shape[-1]
        if torch.is_tensor(x):
            xmin = torch.tensor(self.xmin, device=x.device)[:shape]
            xmax = torch.tensor(self.xmax, device=x.device)[:shape]
        else:
            xmin = self.xmin[:shape]
            xmax = self.xmax[:shape]
        return (x - xmin) / (xmax - xmin) * 2 - 1
            
    def unnormalize(self, y: Any) -> Any:
        shape = y.shape[-1]
        if torch.is_tensor(y):
            xmin = torch.tensor(self.xmin, device=y.device)[:shape]
            xmax = torch.tensor(self.xmax, device=y.device)[:shape]
        else:
            xmin = self.xmin[:shape]
            xmax = self.xmax[:shape]
        return 0.5 * (y + 1.0) * (xmax - xmin) + xmin
    
def make_normalizer(name, xmin_xmax):
    logger.info(f'Making normalizer: {name}')
    return NORMALIZER.get(name)(xmin_xmax)


