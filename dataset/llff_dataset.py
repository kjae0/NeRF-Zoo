import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

def _move_rotation_matrix_axis(poses):
    poses = torch.concat([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], dim=1)
    poses = poses.permute(2, 0, 1)

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral():
    pass

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

class LLFFDataset(Dataset):
    def __init__(self, cfg):
        self.images, self.poses, self.nears, self.fars = self.load_data(cfg)
        
        self.totensor = transforms.ToTensor()
        self.poses = torch.tensor(self.poses, dtype=torch.float32)
        self.nears = torch.tensor(self.nears, dtype=torch.float32)
        self.fars = torch.tensor(self.fars, dtype=torch.float32)
                
        # Batch first
        self.poses = _move_rotation_matrix_axis(self.poses)
        self.nears = self.nears.permute(1, 0)
        self.fars = self.fars.permute(1, 0)
        
        assert len(self.images) == len(self.poses) == len(self.nears) == len(self.fars), \
            'Number of images, poses, nears, and fars are not the same! got {}, {}, {}, {}'.format(
                len(self.images), len(self.poses), len(self.nears), len(self.fars))
            
        if 'boundary_factor' in cfg:
            boundary_factor = 1 / (cfg['boundary_factor'] * self.nears.min())
        else:
            print("Warning! boundary_factor is not provided. Using default value of 0.75")
            
        self.poses[:, :3, 3] *= boundary_factor
        self.nears *= boundary_factor
        self.fars *= boundary_factor
        
        self.c2w = poses_avg(self.poses)
        # TODO recenter, shperify, sprial path

    def _load_pose(self, pose_dir, image_shape, factor=8):
        poses_arr = np.load(pose_dir) # (N_images, 17) 15 for camera pose, 2 for near and far
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]) # (3, 5, N_images)
        nears = poses_arr[:, -2:-1].transpose([1, 0]) # (1, N_images)
        fars = poses_arr[:, -1:].transpose([1, 0]) # (1, N_images)
        
        poses[:2, 4, :] = np.array(image_shape).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] / factor
        
        return poses, nears, fars
    
    def _load_image(self, image_dir):
        images = os.listdir(image_dir)
        images = [i for i in images if i.lower().endswith('.png') \
            or i.lower().endswith('.jpg') \
                or i.lower().endswith('.jpeg')]
        images.sort()
        
        return images

    def load_data(self, base_dir, image_folder='images', image_preload=False):
        images = self._load_image(os.path.join(base_dir, image_folder))
        image_shape = np.array(Image.open(images[0]).convert('RGB')).shape # (H, W, C)
        
        poses, nears, fars = self._load_pose(os.path.join(base_dir, 'poses_bounds.npy'), image_shape)
        
        if image_preload:
            totensor = transforms.ToTensor()
            images = [totensor(Image.open(i).convert('RGB')) for i in images]
            
        return images, poses, self.c2w, nears, fars

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        near = self.nears[idx]
        far = self.fars[idx]
        
        return image, pose, near, far


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test