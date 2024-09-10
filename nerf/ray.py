import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_all_rays(H, W, K, poses):
    assert len(poses.shape) == 3, "Poses must be of shape (B, 3, 4)"
    
    ray_ods = [get_rays(H, W, K, pose) for pose in poses]  # List[Tuple((H, W, 3), (H, W, 3))]
    ray_origins = [ray_od[0] for ray_od in ray_ods]  # List[(H, W, 3)]
    ray_directions = [ray_od[1] for ray_od in ray_ods]  # List[(H, W, 3)]

    ray_origins = torch.stack(ray_origins, 0)  # (B, H, W, 3)
    ray_directions = torch.stack(ray_directions, 0)  # (B, H, W, 3)
    
    B, H, W, _ = ray_origins.shape
    ray_origins = torch.reshape(ray_origins, [B, -1, 3])  # (B, H * W, 3)
    ray_directions = torch.reshape(ray_directions, [B, -1, 3])  # (B, H * W, 3)
    
    # TODO precrop for early training stage.
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

    return ray_origins, ray_directions, coords

def get_rays(H, W, K, c2w):
    """
    Args:
        H : height
        W : width
        K : camera intrinsic parameter
        c2w : camera to world transformation matrix
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    
    ppx = K[0][2]
    ppy = K[1][2]
    focal_x = K[0][0]
    focal_y = K[1][1]
    
    ds = torch.stack([(i-ppx)/focal_x, 
                      -(j-ppy)/focal_y, 
                      -torch.ones_like(i)], -1)
    rays_d = torch.sum(ds[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    
    return rays_o, rays_d

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """
    Code from official nerf impl.
    
    Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map
