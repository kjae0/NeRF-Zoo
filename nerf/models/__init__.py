from nerf.models.nerf import NeRF
from nerf.models.lr_nerf import LowRankNeRF

def build_model(cfg, model_params) -> NeRF:
    if cfg['name'] == 'basic_nerf':
        return NeRF(**model_params)
    elif cfg['name'] == 'lowrank_nerf': 
        return LowRankNeRF(**model_params)
    else:
        raise NotImplementedError(f"Model {cfg['name']} not implemented")
    