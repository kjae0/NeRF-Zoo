import os
import sys
sys.path.append("./")

import torch
import torch.nn as nn

import yaml
import time
import argparse

from torch.utils.data import DataLoader
from nerf.engines import build_engine

# TODO wandb

# OPTIONAL
import tensorboardX

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True, help="Path to the config file")
    parser.add_argument("--enable_writer", action='store_true', default=False, help="Path to the config file")
    args = parser.parse_args()
    
    with open(args.cfg_dir, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    print(cfg)
    
    cfg['dataset']['base_dir'] = os.path.join(cfg['dataset']['base_dir'], cfg['dataset']['object'])
    ckpt_name = f"{args.cfg_dir.split('/')[-1].split('.')[0]}_{time.strftime('%Y%m%d-%H%M%S')}"
    cfg['ckpt_dir'] = os.path.join(cfg['ckpt_dir'], cfg['dataset']['object'], ckpt_name)
    
    if not os.path.exists(cfg['ckpt_dir']):
        os.makedirs(cfg['ckpt_dir'])

        with open(os.path.join(cfg['ckpt_dir'], 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)
        
    if args.enable_writer:
        writer = tensorboardX.SummaryWriter(cfg['ckpt_dir'])
    else:
        writer = None
        
    engine = build_engine(cfg, writer=writer)
    engine.train(cfg)
    