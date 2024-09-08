from nerf.engines.base import BaseEngine
from nerf.models import build_model
from nerf.utils import build_loss_fn, build_optimizer, build_scheduler

class BasicNeRF(BaseEngine):
    def __init__(self, cfg):
        self.model = build_model(cfg)
        self.loss_fn = build_loss_fn(cfg)
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)

    def train_one_epoch(self, data_loader):
        self.model.train()
        for batch in data_loader:
            pass
        
    def train(self, data_loader):
        pass
    
    def evaluate(self, data_loader):
        pass
    