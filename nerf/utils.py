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
        return torch.optim.Adam(params, lr=cfg['optimizer_params']['lr'], betas=cfg['optimizer_params']['betas'])
    elif cfg['name'] == 'rmsprop':
        return torch.optim.RMSprop(params, momentum=0.9, lr=cfg['optimizer_params']['lr'])
    elif cfg['name'] == 'sgd':
        return torch.optim.SGD(params, nesterov=True, momentum=0.9, **cfg['optimizer_params'])
    else:
        raise NotImplementedError(f"Optimizer {cfg['optimizer']} not implemented")
    
def build_scheduler(cfg, optimizer):
    if cfg['name'] == 'none':
        return None
    elif cfg['name'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['scheduler_params']['step_size'], gamma=cfg['scheduler_params']['gamma'])
    elif cfg['name'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler_params']['t_max'], eta_min=cfg['scheduler_params']['eta_min'])
    else:
        raise NotImplementedError(f"Scheduler {cfg['name']} not implemented")

def raw2outputs(raw_rgb, raw_sigma, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, device='cpu'):
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
    
    # Ensure inputs are on the correct device
    raw_rgb = raw_rgb.to(device)
    raw_sigma = raw_sigma.to(device)
    z_vals = z_vals.to(device)
    rays_d = rays_d.to(device)

    # Concatenate raw_rgb and raw_sigma
    raw = torch.cat([raw_rgb, raw_sigma], -1).to(device)
    
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    # Compute distances between consecutive z_vals
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)  # [batch_size, num_rays, num_samples]
    
    # Adjust distances based on ray directions
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Compute RGB values
    rgb = torch.sigmoid(raw[..., :3])  # [batch_size, num_rays, num_samples, 3]
    
    # Add noise to sigma values if specified
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std

    # Compute alpha values from sigma and noise
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [batch_size, num_rays, num_samples]
    
    # Compute weights for volume rendering
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1), device=device), 1.-alpha + 1e-10], -1), -1
    )[:, :, :-1]

    # Compute RGB map (weighted sum of sampled colors)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [batch_size, num_rays, 3]

    # Compute depth map (weighted sum of z values)
    depth_map = torch.sum(weights * z_vals, -1)
    
    # Compute disparity map (inverse of depth map)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))
    
    # Compute accumulated opacity map
    acc_map = torch.sum(weights, -1)

    # Apply white background if specified
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def sample_pdf(bins, weights, N_samples, det=False, pytest=False, device='cpu'):
    # Ensure bins and weights are on the correct device
    bins = bins.to(device)
    weights = weights.to(device)

    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1, device=device), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds, device=device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # Gather cdf and bins values at the sampled indices
    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 3, inds_g)
    bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), 3, inds_g)

    # Calculate sample points based on the inverse CDF
    denom = (cdf_g[...,1] - cdf_g[...,0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[...,0]) / denom
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])

    return samples

def get_learning_rate(optimizer):
    # PyTorch optimizers can have multiple parameter groups, each with its own learning rate.
    # Here, we retrieve the learning rate of the first parameter group.
    for param_group in optimizer.param_groups:
        return param_group['lr']

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]
    
    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d
