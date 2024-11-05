from nerf.engines.base import BaseEngine
from nerf.models import build_model
from src.utils import save_checkpoints, save_images
from src.metrics import calculate_psnr, calculate_mse
from nerf.utils import (build_loss_fn, 
                        build_optimizer, 
                        build_scheduler, 
                        raw2outputs, 
                        sample_pdf, 
                        get_learning_rate,
                        ndc_rays)

from tqdm import tqdm
from torch.utils.data import DataLoader

import os
import torch
import time
import random
import numpy as np

from nerf_datasets import build_dataset

class BasicNeRF(BaseEngine):
    def __init__(self, cfg, writer=None):
        self.nerf_coarse = build_model(cfg['model'], cfg['model']['coarse_model_params'])
        self.nerf_fine = build_model(cfg['model'], cfg['model']['fine_model_params'])
        self.loss_fn = build_loss_fn(cfg['loss'])
        self.optimizer = build_optimizer(cfg['optimizer'], list(self.nerf_coarse.parameters()) + list(self.nerf_fine.parameters())) 
        self.scheduler = build_scheduler(cfg['scheduler'], self.optimizer)
        self.train_dataset, self.test_dataset = build_dataset(cfg['dataset'], test_spiral=cfg['test']['test_spiral'])
        
        total_params = sum(p.numel() for p in self.nerf_fine.parameters())
        trainable_params = sum(p.numel() for p in self.nerf_fine.parameters() if p.requires_grad)

        print(f"[INFO] Total parameters: {total_params}")
        print(f"[INFO] Trainable parameters: {trainable_params}")
        print(f"[INFO] Expected last learning rate: {cfg['optimizer']['optimizer_params']['lr'] * (cfg['scheduler']['scheduler_params']['gamma'] ** int(cfg['train']['num_epochs'] / cfg['scheduler']['scheduler_params']['step_size']))}")
        
        self.device = cfg['device']
        self.verbose = cfg['verbose']
        self.log_interval = cfg['log_interval']
        self.save_interval = cfg['save_interval']        
        self.n_coarse_samples = cfg['train']['n_coarse_samples']
        self.n_fine_samples = cfg['train']['n_fine_samples']
        self.white_bkgd = cfg['train']['white_bkgd']
        self.perturb = cfg['train']['perturb']
        self.raw_noise_std = cfg['train']['raw_noise_std']
        self.ndc = cfg['dataset']['ndc']
        
        # tensorboard writer. optional.
        self.writer = writer
        
        self.nerf_coarse.to(self.device)
        self.nerf_fine.to(self.device)
        self.loss_fn.to(self.device)
        
    def run_network(self, model, pts, viewdirs):
        # pts -> n_rays x n_samples x 3, viewdirs -> B*n_samples x 3
        B, n_rays, n_samples, _ = pts.shape
        viewdirs = torch.concat([viewdirs.unsqueeze(2) for _ in range(n_samples)], dim=2)
        
        # pts -> # [B x n_rays x n_samples x 3], viewdirs -> [B x n_rays x n_samples x 3]
        pts = torch.reshape(pts, [-1, 3]) # [B*n_rays*n_samples x 3]
        viewdirs = torch.reshape(viewdirs, [-1, 3]) # [B*n_rays*n_samples x 3]

        pts = pts.to(self.device)
        viewdirs = viewdirs.to(self.device)
        out_rgb, out_sigma = model(pts, viewdirs)
        
        out_rgb = torch.reshape(out_rgb, [B, n_rays, n_samples, 3])
        out_sigma = torch.reshape(out_sigma, [B, n_rays, n_samples, 1])
        
        return out_rgb, out_sigma

    def train_one_epoch(self, dataloader, hwf=None):
        self.nerf_coarse.train()
        self.nerf_fine.train()
            
        s = time.time()
        si = time.time()
        train_loss_coarse = 0
        train_loss_fine = 0
        n_iters = 0
        
        for iter, (targets, pose, ray_origins, ray_directions, near, far) in enumerate(dataloader):
            ray_origins = ray_origins.to(self.device)
            ray_directions = ray_directions.to(self.device)
            near = near.to(self.device)
            far = far.to(self.device)
            targets = targets.permute(0, 2, 1).to(self.device)
            
            out = self.render(ray_origins, ray_directions, near, far, 
                              perturb=self.perturb, raw_noise_std=self.raw_noise_std, 
                              ndc=self.ndc, hwf=hwf)
            coarse = out['coarse_rgb_map'] # B x n_rays x 3
            fine = out['fine_rgb_map'] # B x n_rays x 3
            
            coarse_loss = self.loss_fn(coarse, targets)
            fine_loss = self.loss_fn(fine, targets)
            loss = coarse_loss + fine_loss
            
            train_loss_coarse += coarse_loss.item()
            train_loss_fine += fine_loss.item()
            n_iters += 1
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.verbose and (iter % self.log_interval == 0):
                print(f"{iter+1} / {len(dataloader)} Loss: ", loss.item(), "Time Elapsed: ", time.time() - si)
                si = time.time()
                
            if self.writer is not None:
                self.writer.add_scalar("coarse NeRF train Loss", coarse_loss.item(), iter)
                self.writer.add_scalar("fine NeRF train Loss", fine_loss.item(), iter)
                
        return {'coarse train loss': train_loss_coarse / n_iters,
                'fine train loss': train_loss_fine / n_iters}, time.time() - s
            
        
    def train(self, cfg):
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        
        train_dataloader = DataLoader(train_dataset, 
                                        batch_size=cfg['train']['batch_size'], 
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=cfg['num_workers'])
    
        H = train_dataset.get_H()
        W = train_dataset.get_W()
        K = train_dataset.get_K()
        focal = train_dataset.get_focal()
        
        for epoch in range(cfg['train']['num_epochs']):
            train_loss_dict, elapsed_time = self.train_one_epoch(train_dataloader, hwf=(H, W, focal))
        
            print(f"[INFO] {epoch+1} / {cfg['train']['num_epochs']} Train Loss", end=" ")
            for k, v in train_loss_dict.items():
                print(f"{k} - {v:6f}", end=" ")
            print(f"Time: {elapsed_time:2f} Lr: {get_learning_rate(self.optimizer)}")
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # evaluate
            if (epoch+1) % cfg['test']['eval_interval'] == 0: 
                metric_dict = {'MSE': calculate_mse, 'PSNR': calculate_psnr}
                perf, preds, gts, elapsed_time = self.evaluate(cfg, test_dataset, metric_dict, hwf=(H, W, focal))

                print(f"\n{epoch+1} / {cfg['train']['num_epochs']} Eval Results")
                for k, v in perf.items():
                    print(f"{k}: {v}")
                print(f"Time: {elapsed_time:2f}\n")
                
                if cfg['test']['save_rendered']:
                    save_images(os.path.join(cfg['ckpt_dir'], "images"), preds, gts)
            
                if self.writer is not None:   
                    # val result
                    for k, v in perf.items():
                        self.writer.add_scalar(k, v, epoch)
                     
                    self.writer.add_image('GT', gts[0], epoch)
                    self.writer.add_image('Pred', preds[0], epoch)
            
            if (epoch+1) % self.save_interval == 0:
                ckpt = {
                    'nerf_coarse': self.nerf_coarse.state_dict(),
                    'nerf_fine': self.nerf_fine.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'random_states': {
                        'torch': torch.get_rng_state(),
                        'numpy': np.random.get_state(),
                        'python': random.getstate()
                    }
                }
                save_checkpoints(ckpt_path=os.path.join(cfg['ckpt_dir'], f"ckpt_{epoch+1}.pt"),
                                 ckpt=ckpt,
                                 epoch=epoch+1,
                                 train_loss=train_loss_dict,
                                 val_loss=None)
    
        
    def render(self, rays_origin, rays_direction, near, far, ndc=False, perturb=0., raw_noise_std=0., hwf=None):
        # rays_origin = rays_origin.view(-1, 3)  # B*n_rays x 3
        # rays_direction = rays_direction.view(-1, 3)  # B*n_rays x 3
        viewdirs = rays_direction
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        
        # viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        viewdirs = viewdirs.float() # B x n_rays x 1
        
        if ndc:
            assert hwf is not None, "HWF is required for NDC rendering."
            # def ndc_rays(H, W, focal, near, rays_o, rays_d):
            H, W, focal = hwf
            
            rays_origin, rays_direction = ndc_rays(H, W, focal, 1., rays_origin, rays_direction)
        
        near = near.unsqueeze(1) * torch.ones_like(rays_direction[...,:1]) # B x n_rays x 1
        far = far.unsqueeze(1) * torch.ones_like(rays_direction[...,:1]) # B x n_rays x 1

        out = self.render_rays(rays_origin, rays_direction, viewdirs, near, far,
                               n_samples=self.n_coarse_samples, n_samples_importance=self.n_fine_samples,
                               white_bkgd=self.white_bkgd, perturb=perturb, raw_noise_std=raw_noise_std)

        return out

    def render_rays(self, rays_origin, rays_direction,
                    viewdirs, near, far, n_samples, n_samples_importance,
                    perturb=0., white_bkgd=False, raw_noise_std=0.):
        
        ret = {}
        # check flat
        # 1. check number of rays
        B, n_rays, _ = rays_origin.shape # B x n_rays(sample ray per image) x 3
        # 2. get rays origin and direction
        # 3. get near far
        
        # 4. set t for stratified sampling
        t_vals = torch.linspace(0., 1., steps=n_samples).to(self.device)

        # 5. sample points (coarse)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = z_vals.expand([B, n_rays, n_samples])
        # z_vals = torch.stack([z_vals for _ in range(n_rays)], dim=0) # [n_rays, n_samples]

        # add perturbation
        if perturb > 0.:
            mids = (z_vals[...,1:] + z_vals[...,:-1]) / 2
            uppers = torch.concat([mids, z_vals[...,-1:]], -1)
            lowers = torch.concat([z_vals[...,:1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=self.device) * perturb
            z_vals = lowers + (uppers - lowers) * t_rand

        sampled_points = rays_origin[...,None,:] + rays_direction[...,None,:] * z_vals[...,:,None] # [n_rays, n_samples, 3]
        
        # 6. compute coarse output (as a pdf) + raw2outputs
        raw_rgb, raw_sigma = self.run_network(self.nerf_coarse,
                                            sampled_points, 
                                            viewdirs)
        # raw_rgb = raw_rgb.cpu()
        # raw_sigma = raw_sigma.cpu()
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_rgb, raw_sigma, 
                                                                     z_vals, rays_direction, 
                                                                     raw_noise_std, white_bkgd, device=self.device)
        ret['coarse_rgb_map'] = rgb_map
        ret['coarse_disp_map'] = disp_map
        ret['coarse_acc_map'] = acc_map
        ret['coarse_weights'] = weights
        ret['coarse_depth_map'] = depth_map
        
        # 7. sample from pdf (importance sampling)
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], n_samples_importance, det=(perturb==0.), device=self.device)
        z_samples = z_samples.detach() # [batch_size, n_rays, n_samples_importance]

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        sampled_points = rays_origin[...,None,:] + rays_direction[...,None,:] * z_vals[...,:,None] # [batch_size, n_rays, n_samples + n_importance, 3]
        
        # 8. compute fine output        
        raw_rgb, raw_sigma = self.run_network(self.nerf_fine,
                                            sampled_points, 
                                            viewdirs)
        # raw_rgb = raw_rgb.cpu()
        # raw_sigma = raw_sigma.cpu()
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_rgb, raw_sigma, z_vals, 
                                                                     rays_direction, raw_noise_std, 
                                                                     white_bkgd, device=self.device)
        ret['fine_rgb_map'] = rgb_map
        ret['fine_disp_map'] = disp_map
        ret['fine_acc_map'] = acc_map
        ret['fine_weights'] = weights
        ret['fine_depth_map'] = depth_map
        
        return ret

    
    def render_spiral(self, dataset, batch_size=10, verbose=True, n_views=120, n_rots=2, render_train=False, test=None, near=None, far=None):
        if render_train:
            ray_origins, ray_directions, coords = dataset.ray_origins, dataset.ray_directions, dataset.coords
            near, far = dataset.nears, dataset.fars
        else: 
            ray_origins, ray_directions, coords = dataset.get_spiral_rays()
            if near:
                near = torch.FloatTensor([near for _ in range(ray_directions.shape[0])]).unsqueeze(-1) # B x 1
                far = torch.FloatTensor([far for _ in range(ray_directions.shape[0])]).unsqueeze(-1) # B x 1
            else:
                near = torch.FloatTensor([dataset.nears[0] for _ in range(ray_directions.shape[0])]).unsqueeze(-1) # B x 1
                far = torch.FloatTensor([dataset.fars[0] for _ in range(ray_directions.shape[0])]).unsqueeze(-1) # B x 1
        # ray_origins -> n_views x (H*W) x 3
        # ray_directions -> n_views x (H*W) x 3
        # coords -> (H*W) x 2
        
            # Move testing data to GPU
        if test:
            ray_origins = ray_origins[:test]
            ray_directions = ray_directions[:test]
            near = near[:test]
            far = far[:test]
            
        H = dataset.get_H()
        W = dataset.get_W()
        K = dataset.get_K()
        focal = dataset.get_focal()

        with torch.no_grad():
            images = None
            print(f"{ray_directions.shape[0]} images will be rendered.")

            self.nerf_coarse.eval()
            self.nerf_fine.eval()
            
            rendered = []
            
            if verbose:
                frames = tqdm(range(0, ray_directions.shape[0], 1), ncols=100, desc="Rendering...")
            else:
                frames = range(0, ray_directions.shape[0], 1)
            
            for i in frames:
                # TODO check near far
                out = []
                for j in range(0, ray_directions.shape[1], batch_size):
                    ret = self.render(ray_origins[i:i+1, j:j+batch_size], ray_directions[i:i+1, j:j+batch_size], near[i:i+1], far[i:i+1], 
                                      perturb=0., raw_noise_std=0., ndc=self.ndc, hwf=(H, W, focal))
                    out.append(ret['fine_rgb_map'].cpu())
                rendered.append(torch.concat(out, dim=1))
                
                # ret = self.render(ray_origins[i:i+batch_size], ray_directions[i:i+batch_size], near[i:i+batch_size], far[i:i+batch_size])
                # out.append([ret['fine_rgb_map'].cpu(), ret['fine_disp_map'].cpu(), ret['fine_acc_map'].cpu()])
                
            self.nerf_coarse.train()
            self.nerf_fine.train()
            
            # print('Done rendering', testsavedir)
            # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)


        # if savedir is not None:
        #         rgb8 = to8b(rgbs[-1])
        #         filename = os.path.join(savedir, '{:03d}.png'.format(i))
        #         imageio.imwrite(filename, rgb8)


        # rgbs = np.stack(rgbs, 0)
        # disps = np.stack(disps, 0)

        return rendered
    
    def load_state_dict(self, state_dict):
        ckpt = state_dict['checkpoint']
        self.nerf_coarse.load_state_dict(ckpt['nerf_coarse'])
        self.nerf_fine.load_state_dict(ckpt['nerf_fine'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        print(f"[INFO] Loaded state dict from epoch {state_dict['epoch']} successfully.")
        
    def evaluate(self, cfg, test_dataset, metric_dict, hwf=None):
        if hwf == None:
            H = test_dataset.get_H()
            W = test_dataset.get_W()
            K = test_dataset.get_K()
            focal = test_dataset.get_focal()
            hwf = (H, W, focal)
        else:
            H, W, focal = hwf
            
        s = time.time()
        
        self.nerf_coarse.eval()
        self.nerf_fine.eval()
        
        gts = []
        preds = []
        
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), total=len(test_dataset), ncols=100, desc="Evaluating..."):
                target, pose, ray_origin, ray_direction, near, far = test_dataset.__getitem__(i, eval=True)
                
                target = target.view(3, -1) # 3 x n_rays
                ray_origin = ray_origin.to(self.device)
                ray_direction = ray_direction.to(self.device)
                near = near.to(self.device)
                far = far.to(self.device)
                
                pred = []
                for j in range(0, target.shape[-1], cfg['test']['chunk_size']):
                    ray_origin_chunk = ray_origin[j:j+cfg['test']['chunk_size']].unsqueeze(0)
                    ray_direction_chunk = ray_direction[j:j+cfg['test']['chunk_size']].unsqueeze(0)

                    out = self.render(ray_origin_chunk, ray_direction_chunk, near, far,   
                                        perturb=0., raw_noise_std=0., ndc=self.ndc, hwf=hwf)
                
                    fine = out['fine_rgb_map'].cpu() # 1 x chunk_size x 3
                    pred.append(fine)
                    
                pred = torch.concat(pred, dim=1) # 1 x n_rays x 3
                gts.append(target.view(3, H, W))
                preds.append(pred.view(H, W, 3).permute(2, 0, 1))

        perf = {}
        for k, v in metric_dict.items():
            if k not in perf:
                perf[k] = 0
            for i in range(len(gts)):
                perf[k] += v(gts[i], preds[i])
        
        for k, v in perf.items():
            perf[k] /= len(gts)
            
        elapsed_time = time.time() - s
        
        return perf, preds, gts, elapsed_time
    