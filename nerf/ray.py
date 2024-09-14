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

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    