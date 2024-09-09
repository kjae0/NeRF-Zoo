from nerf.engines.base import BaseEngine
from nerf.models import build_model
from nerf.utils import build_loss_fn, build_optimizer, build_scheduler

from tqdm import tqdm
import torch

class BasicNeRF(BaseEngine):
    def __init__(self, cfg):
        self.model = build_model(cfg)
        self.loss_fn = build_loss_fn(cfg)
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.near = cfg.near
        self.far = cfg.far
        self.verbose = cfg.verbose

    def train_one_epoch(self, data_loader):
        self.model.train()
        
        if self.verbose:
            data_loader = tqdm(data_loader, ncols=100, total=len(data_loader), desc="training...")
            
        for image, pose, c2w, near, far in data_loader:
            image = image.to(self.device)
            
            
            
        
    def train(self, data_loader, H, W, K=None, focal=None):
        if K is None:
            if focal is None:
                raise ValueError("focal length must be provided")
            
            K = torch.tensor([[focal, 0, W // 2], [0, focal, H // 2], [0, 0, 1]])
            
        pass
    
    def evaluate(self, data_loader):
        pass
    