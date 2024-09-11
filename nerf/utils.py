import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def build_loss_fn(cfg):
    if cfg['name'] == 'l2':
        return nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function {cfg['loss_fn']} not implemented")
    
def build_optimizer(cfg, params):
    if cfg['name'] == 'adam':
        return torch.optim.Adam(params, **cfg['optimizer_params'])
    else:
        raise NotImplementedError(f"Optimizer {cfg['optimizer']} not implemented")
    
def build_scheduler(cfg, optimizer):
    if cfg['name'] == 'none':
        return None
    elif cfg['name'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg['scheduler_params'])
    elif cfg['name'] == 'multisteplr':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **cfg['scheduler_params'])
    else:
        raise NotImplementedError(f"Scheduler {cfg['name']} not implemented")

def raw2outputs(raw_rgb, raw_sigma, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw_rgb: [batch_size, num_rays, num_samples along ray, 3]. Prediction from model.
        raw_sigma: [batch_size,  num_rays, num_samples along ray, 1]. Prediction from model.
        z_vals: [batch_size, num_rays, num_samples along ray]. Integration time.
        rays_d: [batch_size, num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [batch_size, num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [batch_size, num_rays]. Disparity map. Inverse of depth map.
        acc_map: [batch_size, num_rays]. Sum of weights along each ray.
        weights: [batch_size, num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [batch_size, num_rays]. Estimated distance to object.
    """
    raw = torch.concat([raw_rgb, raw_sigma], -1)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [B, N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [B, N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 3, inds_g)
    bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), 3, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def get_learning_rate(optimizer):
    # PyTorch optimizers can have multiple parameter groups, each with its own learning rate.
    # Here, we retrieve the learning rate of the first parameter group.
    for param_group in optimizer.param_groups:
        return param_group['lr']
    