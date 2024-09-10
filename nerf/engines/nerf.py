from nerf.engines.base import BaseEngine
from nerf.models import build_model
from nerf.ray import get_rays
from nerf.utils import build_loss_fn, build_optimizer, build_scheduler, raw2outputs, sample_pdf
from src.utils import save_checkpoints

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import time

class BasicNeRF(BaseEngine):
    def __init__(self, cfg, writer=None):
        self.nerf_coarse = build_model(cfg['model'], cfg['model']['coarse_model_params'])
        self.nerf_fine = build_model(cfg['model'], cfg['model']['fine_model_params'])
        self.loss_fn = build_loss_fn(cfg['loss'])
        self.optimizer = build_optimizer(cfg['optimizer'], list(self.nerf_coarse.parameters()) + list(self.nerf_fine.parameters())) 
        self.scheduler = build_scheduler(cfg['scheduler'], self.optimizer)
        
        self.device = cfg['device']
        self.verbose = cfg['verbose']
        self.log_interval = cfg['log_interval']
        self.save_interval = cfg['save_interval']
        
        self.n_coarse_samples = cfg['train']['n_coarse_samples']
        self.n_fine_samples = cfg['train']['n_fine_samples']
        self.white_bkgd = cfg['train']['white_bkgd']
        
        # tensorboard writer. optional.
        self.writer = writer
        
        self.nerf_coarse.to(self.device)
        self.nerf_fine.to(self.device)
        self.loss_fn.to(self.device)
        
    def render(self, rays_origin, rays_direction, near, far):
        # TODO validate ray and camera parameters
        rays_origin = rays_origin.view(-1, 3)  # B*n_rays x 3
        rays_direction = rays_direction.view(-1, 3)  # B*n_rays x 3
        
        near = near * torch.ones_like(rays_direction[...,:1])
        far = far * torch.ones_like(rays_direction[...,:1])
        
        viewdirs = rays_direction
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        
        out = self.render_rays(rays_origin, rays_direction, viewdirs, near, far,
                               n_samples=self.n_coarse_samples, n_samples_importance=self.n_fine_samples,
                               white_bkgd=self.white_bkgd)

        return out

    def render_rays(self,
                    rays_origin,
                    rays_direction,
                    viewdirs,
                    near, far,
                    n_samples,
                    n_samples_importance,
                    perturb=0.,
                    white_bkgd=False,
                    raw_noise_std=0.):
        
        ret = {}
        # check flat
        # 1. check number of rays
        n_rays = rays_origin.shape[0] # B*n_rays(sample ray per image) x 3
        # 2. get rays origin and direction
        # 3. get near far
        
        # 4. set t for stratified sampling
        t_vals = torch.linspace(0., 1., steps=n_samples)

        # 5. sample points (coarse)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = z_vals.expand([n_rays, n_samples])
        # z_vals = torch.stack([z_vals for _ in range(n_rays)], dim=0) # [n_rays, n_samples]
        
        # add perturbation
        mids = (z_vals[...,1:] + z_vals[...,:-1]) / 2
        uppers = torch.concat([mids, z_vals[...,-1:]], -1)
        lowers = torch.concat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lowers + (uppers - lowers) * t_rand

        sampled_points = rays_origin[...,None,:] + rays_direction[...,None,:] * z_vals[...,:,None] # [n_rays, n_samples, 3]
        
        # 6. compute coarse output (as a pdf) + raw2outputs
        raw_rgb, raw_sigma = self.run_network(self.nerf_coarse, 
                                              sampled_points, 
                                              viewdirs)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_rgb, raw_sigma, 
                                                                     z_vals, rays_direction, 
                                                                     raw_noise_std, white_bkgd)
        ret['coarse_rgb_map'] = rgb_map
        ret['coarse_disp_map'] = disp_map
        ret['coarse_acc_map'] = acc_map
        ret['coarse_weights'] = weights
        ret['coarse_depth_map'] = depth_map
        
        # 7. sample from pdf (importance sampling)
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], n_samples_importance, det=(perturb==0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        sampled_points = rays_origin[...,None,:] + rays_direction[...,None,:] * z_vals[...,:,None] # [n_rays, n_samples + n_importance, 3]
        
        # 8. compute fine output        
        raw_rgb, raw_sigma = self.run_network(self.nerf_fine,
                                              sampled_points, 
                                              viewdirs)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_rgb, raw_sigma, z_vals, rays_direction, raw_noise_std, white_bkgd)
        ret['fine_rgb_map'] = rgb_map
        ret['fine_disp_map'] = disp_map
        ret['fine_acc_map'] = acc_map
        ret['fine_weights'] = weights
        ret['fine_depth_map'] = depth_map
        
        return ret

    def run_network(self, model, pts, viewdirs):
        B = pts.shape[0]
        viewdirs = torch.stack([viewdirs for _ in range(B)], dim=0)
        
        # pts -> [B, N, 3], viewdirs -> [B, N, 3]
        pts = torch.reshape(pts, [-1, 3]) # [B*N, 3]
        viewdirs = torch.reshape(viewdirs, [-1, 3]) # [B*N, 3]
        
        pts = pts.to(self.device)
        viewdirs = viewdirs.to(self.device)
        out_rgb, out_sigma = model(pts, viewdirs)
        
        out_rgb = torch.reshape(out_rgb, [B, N, 3])
        out_sigma = torch.reshape(out_sigma, [B, N, 1])
        
        return out_rgb, out_sigma

    def train_one_epoch(self, dataloader):
        self.nerf_coarse.train()
        self.nerf_fine.train()
        
        # if self.verbose:
        #     dataloader = tqdm(dataloader, ncols=100, total=len(dataloader), desc="training...")
            
        s = time.time()
        si = time.time()
        for iter, (targets, pose, ray_origins, ray_directions, near, far) in enumerate(dataloader):
            # B, H, W, _ = image.shape
            # print(targets.shape, pose.shape, ray_origins.shape, ray_directions.shape, near.shape, far.shape)
            targets = targets.to(self.device)
            # pose = pose.to(self.device)
            # near = near.to(self.device)
            # far = far.to(self.device)
            # ray_origins = ray_origins.to(self.device)
            # ray_directions = ray_directions.to(self.device) # B x n_rays x 3
            
            out = self.render(ray_origins, ray_directions, near, far)
            coarse = out['coarse_rgb_map'] # B * n_rays x 3
            fine = out['fine_rgb_map'] # B * n_rays x 3
            targets = targets.view(-1, 3) # B * n_rays x 3
            
            new_view = (coarse + fine) / 2
            loss = self.loss_fn(new_view, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.verbose and (iter % self.log_interval == 0):
                print(f"{iter+1} / {len(dataloader)} Loss: ", loss.item(), "Time Elapsed: ", time.time() - si)
                si = time.time()
                
        print("Time: ", time.time() - s)
            
        
    def train(self, cfg, train_dataset):
        cfg['train']['num_epochs'] = int(cfg['train']['num_iterations'] / len(train_dataset) * cfg['train']['batch_size'])
        self.epoch = cfg['train']['num_epochs']
        
        train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg['train']['batch_size'], 
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=cfg['num_workers'])
    
        H = train_dataset.get_H()
        W = train_dataset.get_W()
        K = train_dataset.get_K()
        focal = train_dataset.get_focal()
        
        for epoch in range(self.epoch):
            print("======================= Epoch {} / {} =======================".format(epoch+1, self.epoch))
            train_loss, elapsed_time = self.train_one_epoch(train_dataloader)
        
            print(f"{epoch+1} Train Loss: ", train_loss, "Time: ", elapsed_time)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # TODO tensorboard logging...
            if self.writer is not None:
                self.writer.add_scalar("Train Loss", train_loss, epoch)
            
            if (epoch+1) % self.save_interval == 0:
                save_checkpoints(self.nerf_coarse, self.nerf_fine, epoch, cfg['checkpoint_dir'])
    
    def evaluate(self, dataloader):
        pass
