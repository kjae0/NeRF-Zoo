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
from nerf_datasets import build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True, help="Path to the config file")
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
    
    # Load the dataset
    # TODO test dataset for validation
    train_dataset, test_dataset = build_dataset(cfg['dataset'], test_spiral=cfg['test']['test_spiral'])
    
    # TODO DDP / DataParallel
    engine = build_engine(cfg)
    engine.train(cfg, train_dataset, test_dataset)
    
    