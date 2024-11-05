import os
import cv2
import json
import torch
import imageio
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from nerf_datasets.ray import get_rays, get_all_rays  # Ensure you have these utility functions

# Follows official NeRF implementation

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()



def load_blender_data(base_dir="./data/nerf_synthetic/lego", half_res=False, testskip=30):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(base_dir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(base_dir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, [H, W, focal], i_split


class BlenderDataset(Dataset):
    def __init__(self, cfg, dset_type):
        self.base_dir = cfg['base_dir']
        self.object = cfg['object']
        self.resize_factor = cfg['factor']
        self.image_preload = cfg['image_preload']
        self.dset_type = dset_type
        self.meta = None
        self.image_size = None
        self.image_transform = None
        
        if cfg['white_bkgd']:
            img_alpha_transform = lambda img: img[:3]*img[-1:] + (1.-img[-1:])
        else:
            img_alpha_transform = lambda img: img[:3]
    
        with open(os.path.join(self.base_dir, 'transforms_{}.json'.format(dset_type)), 'r') as f:
            self.meta = json.load(f)
        
        if dset_type == 'test':
            skip = 10
        else:
            skip = 1
        
        self.images = self._load_image(self.meta['frames'][::skip])
        if self.image_preload:
            self.images = [Image.open(img_path) for img_path in self.images]
            W, H = self.images[0].size[:2]
            H = H // self.resize_factor
            W = W // self.resize_factor
            self.image_size = (H, W)
            self.image_transform = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor(),
                img_alpha_transform
            ])
            self.images = [self.image_transform(img) for img in self.images]
            
        else:
            img = Image.open(self.images[0])
            W, H = img.size[:2]
            H = H // self.resize_factor
            W = W // self.resize_factor
            self.image_size = (H, W)
            self.image_transform = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor(),
                img_alpha_transform
            ])
            
        self.poses = self._load_pose(self.meta['frames'][::skip])    # (N, 4, 4)
        self.poses = self.poses[:, :3, :]    # (N, 3, 4)
        self.H = H
        self.W = W
        self.camera_angle_x = float(self.meta['camera_angle_x'])
        self.focal = 0.5 * W / np.tan(0.5 * self.camera_angle_x)
        self.render_poses = torch.stack([self._pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

        # Near and Far bounds
        self.near = 2.
        self.far = 6.
        self.nears = torch.full((len(self.images),), self.near, dtype=torch.float32).unsqueeze(1)
        self.fars = torch.full((len(self.images),), self.far, dtype=torch.float32).unsqueeze(1)
        
        self.ray_origins, self.ray_directions, self.coords = get_all_rays(self.H, self.W, self.get_K(), self.poses)
        
        self.n_sample_rays = cfg.get('n_sample_rays', None)
        if self.n_sample_rays == None:
            print(f"Warning! n_sample_rays is not provided. Using all rays.")
        
    def _load_pose(self, frames):
        poses = []
        for frame in frames:
            pose = torch.FloatTensor(frame['transform_matrix'])
            poses.append(pose)
        
        return torch.stack(poses, dim=0)
    
    def _load_image(self, frames):
        images = []
        for frame in frames:
            img_path = os.path.join(self.base_dir, self.dset_type, frame['file_path'].split("/")[-1] + '.png')
            images.append(img_path)
        
        return images
    
    def _pose_spherical(self, theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        
        return c2w
    
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
        
    def get_spiral_rays(self):
        ray_origins, ray_directions, coords = get_all_rays(self.H, self.W, self.get_K(), self.render_poses)
        
        return ray_origins, ray_directions, coords, 
    
    