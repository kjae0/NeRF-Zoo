from nerf.engines.base import BaseEngine
from nerf.engines.nerf import BasicNeRF

def build_engine(cfg) -> BaseEngine:
    if cfg['model']['name'] == 'basic_nerf':
        return BasicNeRF(cfg)
    else:
        raise NotImplementedError(f"Model {cfg['model']} not implemented")