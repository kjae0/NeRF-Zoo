import sys
sys.path.append("./")

from nerf_datasets.llff_dataset import LLFFDataset
from nerf_datasets.blender_dataset import BlenderDataset
from nerf.models import nerf
from nerf.models import build_model
from nerf.engines import build_engine
from src.utils import images_to_video

import cv2
import os
import yaml
import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dset_type", type=str, default="llff")
    
    args = parser.parse_args()
    
    with open(os.path.join(args.ckpt_dir, "config.yaml"), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['device'] = args.device
        
    cfg['dataset']['base_dir'] = '/home/diya/Public/Image2Smiles/jy/NeRF-Zoo/data/nerf_llff_data/fern'
    
    if args.dset_type == 'llff':
        dset = LLFFDataset(cfg['dataset'])
    elif args.dset_type == 'blender':
        dset = BlenderDataset(cfg['dataset'], 'test')
        
    H, W = dset.get_H(), dset.get_W()

    if args.ckpt_name:
        print(f"[INFO] Loading {args.ckpt_name} from {args.ckpt_dir}")
        state_dict = torch.load(os.path.join(args.ckpt_dir, args.ckpt_name), map_location=args.device)        
    else:
        ckpt_names = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pt')]
        ckpt_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"[INFO] Loading {ckpt_names[-1]} from {args.ckpt_dir}")
        state_dict = torch.load(os.path.join(args.ckpt_dir, ckpt_names[-1]), map_location=args.device)

    model = build_engine(cfg)
    model.load_state_dict(state_dict)
    print("[INFO] Model loaded")

    # TODO verbose set
    out = model.render_spiral(dset, batch_size=8192, render_train=False, verbose=True)

    images = [o.view(-1, H, W, 3)[0] for o in out]
    output_video_path = args.out_dir
    images_to_video(images, output_video_path, fps=30)
