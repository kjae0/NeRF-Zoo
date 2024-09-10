import os
import sys
sys.path.append("/home/diya/Public/Image2Smiles/jy/NeRF-Zoo")

import torch
import torch.nn as nn

import yaml
import argparse

from torch.utils.data import DataLoader
from nerf.engines import build_engine
from datasets import build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    
    with open(args.cfg_dir, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    print(cfg)
    cfg['dataset']['base_dir'] = os.path.join(cfg['dataset']['base_dir'], cfg['dataset']['object'])
    
    # Load the dataset
    # TODO test dataset for validation
    train_dataset = build_dataset(cfg['dataset'], test_spiral=cfg['test']['test_spiral'])
    
    # TODO DDP / DataParallel
    engine = build_engine(cfg)
    engine.train(cfg, train_dataset)
    
    