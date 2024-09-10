from nerf.models.nerf import NeRF

def build_model(cfg, model_params) -> NeRF:
    if cfg['name'] == 'basic_nerf':
        return NeRF(**model_params)
    else:
        raise NotImplementedError(f"Model {cfg['name']} not implemented")
    