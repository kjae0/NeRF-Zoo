from nerf.engines.base import BaseEngine
from nerf.engines.nerf import BasicNeRF

def build_engine(cfg, **kwargs) -> BaseEngine:
    # TODO if add more engines
    if True:
        return BasicNeRF(cfg, **kwargs)
    else:
        raise NotImplementedError(f"Model {cfg['model']} not implemented")