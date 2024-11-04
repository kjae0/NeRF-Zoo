import os
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from nerf_datasets.utils import _move_rotation_matrix_axis, poses_avg, normalize, render_path_spiral, recenter_poses
from nerf.ray import get_rays, get_all_rays


class LLFFDataset(Dataset):
    def __init__(self, cfg):
        self.images, self.poses, self.nears, self.fars = self.load_data(base_dir=cfg['base_dir'], 
                                                                        image_folder=cfg['image_folder'], 
                                                                        factor=cfg['factor'], 
                                                                        image_preload=cfg['image_preload'])
        
        self.image_preload = cfg['image_preload']
        if not self.image_preload:
            image_shape = np.array(Image.open(self.images[0]).convert('RGB')).shape # (H, W, C)
            self.image_transform = transforms.Compose([
                transforms.Resize((image_shape[0] // cfg['factor'], image_shape[1] // cfg['factor'])),
                transforms.ToTensor()
            ])
                
        # Batch first
        self.poses = np.concatenate([self.poses[:, 1:2, :], -self.poses[:, 0:1, :], self.poses[:, 2:, :]], 1)
        self.poses = np.moveaxis(self.poses, -1, 0).astype(np.float32)
        self.fars = np.moveaxis(self.fars, -1, 0).astype(np.float32)
        self.nears = np.moveaxis(self.nears, -1, 0).astype(np.float32) 
        
        assert len(self.images) == len(self.poses) == len(self.nears) == len(self.fars), \
            'Number of images, poses, nears, and fars are not the same! got {}, {}, {}, {}'.format(
                len(self.images), len(self.poses), len(self.nears), len(self.fars))
            
        if 'boundary_factor' in cfg:
            boundary_factor = cfg['boundary_factor']
            scale = 1 / (boundary_factor * self.nears.min())
        else:
            print("Warning! boundary_factor is not provided. No scaling will be applied.")
            scale = 1
            
        self.poses[:, :3, 3] *= scale
        self.nears *= scale
        self.fars *= scale
        
        self.poses = recenter_poses(self.poses)
        
        self.poses = torch.tensor(self.poses, dtype=torch.float32)
        self.nears = torch.tensor(self.nears, dtype=torch.float32)
        self.fars = torch.tensor(self.fars, dtype=torch.float32)
        
        self.c2w = poses_avg(self.poses)
        n_views = 120
        n_rots = 2
        path_zflat = False
        self.spiral_poses = self.get_spiral_poses(n_views, n_rots, path_zflat)[:, :3, :4]
        self.spiral_poses = torch.tensor(self.spiral_poses, dtype=torch.float32)
        
        if cfg['ndc']:
            self.nears = np.zeros_like(self.nears)
            self.fars = np.ones_like(self.fars)
        else:
            self.nears = (self.nears.min() * .9) * np.ones_like(self.nears)
            self.fars = (self.fars.max() * 1.) * np.ones_like(self.fars)
        self.nears = torch.tensor(self.nears, dtype=torch.float32)
        self.fars = torch.tensor(self.fars, dtype=torch.float32)
        
        self.hwf = self.poses[0,:3,-1]
        self.H = int(self.hwf[0].item())
        self.W = int(self.hwf[1].item())
        self.focal = self.hwf[2].item()
        self.poses = self.poses[:,:3,:4]
        
        # TODO ray_preload for too much images
        self.ray_origins, self.ray_directions, self.coords = get_all_rays(self.H, self.W, self.get_K(), self.poses)
        
        self.n_sample_rays = cfg.get('n_sample_rays', None)
        if self.n_sample_rays == None:
            print(f"Warning! n_sample_rays is not provided. Using all rays.")
        # TODO recenter, shperify, sprial path        
        
    def get_H(self):
        return self.H
    
    def get_W(self):
        return self.W
    
    def get_K(self):
        K = np.array([
            [self.focal,    0,          0.5*self.W],
            [0,             self.focal, 0.5*self.H],
            [0,             0,          1]
        ])
        return K
    
    def get_focal(self):
        return self.focal

    def _load_pose(self, pose_dir, image_shape, factor=8, recenter=True):
        poses_arr = np.load(pose_dir) # (N_images, 17) 15 for camera pose, 2 for near and far
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]) # (3, 5, N_images)
        bounds = poses_arr[:, -2:].transpose([1, 0]) # (2, N_images) 
        nears = bounds[:1, :] # (1, N_images)
        fars = bounds[1:2, :] # (1, N_images)
            
        poses[:2, 4, :] = np.array(image_shape).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] / factor

        return poses, nears, fars
    
    def _load_image(self, image_dir):
        images = os.listdir(image_dir)
        images = [os.path.join(image_dir, i) for i in images if i.lower().endswith('.png') \
            or i.lower().endswith('.jpg') \
                or i.lower().endswith('.jpeg')]
        images.sort()
        
        return images

    def load_data(self, base_dir, recenter=True, factor=1, image_folder='images', image_preload=False, verbose=False):
        images = self._load_image(os.path.join(base_dir, image_folder))        
        image_shape = np.array(Image.open(images[0]).convert('RGB')).shape # (H, W, C)
        image_shape = (image_shape[0] // factor, image_shape[1] // factor)
        poses, nears, fars = self._load_pose(os.path.join(base_dir, 'poses_bounds.npy'), image_shape, factor=factor, recenter=recenter)
        
        if image_preload:
            image_transform = transforms.Compose([
                transforms.Resize((image_shape[0], image_shape[1])),
                transforms.ToTensor()
            ])
            images_loaded = []
            
            if verbose:
                images = tqdm(images, ncols=100, desc="Preloading images...", total=len(images))
                
            for i in images:
                images_loaded.append(image_transform(Image.open(i).convert('RGB')))
            # images = [image_transform(Image.open(i).convert('RGB')) for i in images]
            images = images_loaded
            
        return images, poses, nears, fars

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, eval=False):
        if self.image_preload:
            image = self.images[idx]
        else:            
            image = self.image_transform(Image.open(self.images[idx]).convert('RGB'))
        pose = self.poses[idx]
        near = self.nears[idx]
        far = self.fars[idx]
        ray_origin = self.ray_origins[idx]
        ray_directions = self.ray_directions[idx]
        coords = self.coords
        
        if self.n_sample_rays is not None and not eval:
            indices = torch.randperm(ray_origin.shape[0])[:self.n_sample_rays]
            ray_origin = ray_origin[indices]
            ray_directions = ray_directions[indices]
            coords = coords[indices]
        # print(image.shape, coords[:, 0].max(), coords[:, 1].max())
        # print(image.shape)
        targets = image[:, coords[:,0].long(), coords[:,1].long()]

        return targets, pose, ray_origin, ray_directions, near, far     

    def get_spiral_poses(self, n_views=120, n_rots=2, path_zflat=False):
        ## Get spiral
        # Get average pose
        up = normalize(self.poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = self.nears.min()*.9, self.fars.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = self.poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 70, 0)
        c2w_path = self.c2w
#         if path_zflat:
# #             zloc = np.percentile(tt, 10, 0)[2]
#             zloc = -close_depth * .1
#             c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
#             rads[2] = 0.
#             N_rots = 1
#             N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=n_rots, N=n_views)
        render_poses = np.array(render_poses).astype(np.float32)
        
        return render_poses
    
    def get_spiral_rays(self):
        ray_origins, ray_directions, coords = get_all_rays(self.H, self.W, self.get_K(), self.spiral_poses)
        
        return ray_origins, ray_directions, coords, 
    