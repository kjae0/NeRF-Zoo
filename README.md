# NeRF Zoo

<strong> Under construction... </strong> <br>
Pytorch implementation of NeRF(Neural Radiance Field) models.<br>
Pretrained weights are provided through GoogleDrive.

CPU bottleneck existing in original implementation is removed! No more bottleneck for multiple run in same time.<br>
This repository aims to simplify NeRF, making it easily customizable and highly extensible.<br>

## Structure
~~~
NeRF-Zoo
    ├── ckpts // checkpoints
    ├── configs
    │   ├── blender
    │   └── llff
    ├── data // datasets
    ├── nerf
    │   ├── engines
    │   ├── models
    │   └── nn
    ├── nerf_datasets
    │   ├── blender_dataset.py
    │   ├── llff_dataset.py
    │   └── ray.py
    └── src
        ├── metrics.py
        ├── render_sprial.py
        ├── test.ipynb
        ├── train.py
        └── utils.py
~~~

~~~
BasicNeRF Class Structure
    ├── __init__: Initializes models, datasets, and training configurations.
    ├── run_network: Runs the model network and reshapes the output.
    ├── train_one_epoch: Conducts a single epoch of training with data loader.
    ├── train: Manages the overall training loop and evaluates periodically.
    ├── render: Computes the rendering of rays using coarse and fine sampling.
    ├── render_rays: Processes ray sampling, applies coarse and fine NeRF models.
    ├── render_spiral: Renders an image sequence along a spiral trajectory for visualization.
    ├── load_state_dict: Loads a checkpoint state dictionary into the models and optimizer.
    └── evaluate: Evaluates model performance on test data, computing metrics like PSNR and MSE.
~~~

## TODO
- refactoring
- other nerf family

## Pretrained Weights
| Model Name | Dataset | PSNR | SSIM | Weights |
|-|-|-|-|-|
| NeRF | LLFF (Fern) | 28.02 | Nan | [Download](https://drive.google.com/file/d/1Z6UcMTHRz9CoycvbDkbHPmdDgaYlKHlc/view?usp=drive_link) |
| NeRF | LLFF (T-Rex) | 31.12 | Nan | [Download](https://drive.google.com/file/d/1wnweWR3EJg2g-FD7_DoMmc5r3kPJY05t/view?usp=drive_link) |
| NeRF | LLFF (Flower) | 27.87 | Nan | [Download](https://drive.google.com/file/d/1J2EwG785t0Ws1Ko18RDWC80e9W8iG-Uu/view?usp=drive_link) |

Coming soon!

## Examples
### <center> NeRF </center>

| Fern | Flower | T-Rex |
|-|-|-|
|  ![fern_nerf](https://github.com/user-attachments/assets/558b9603-6fbc-4fc4-9f74-de1c4d3434dd)| ![flower_nerf](https://github.com/user-attachments/assets/fbdcef67-2034-4f78-b8ff-630b4fe76f27) | ![trex_nerf](https://github.com/user-attachments/assets/ce46287e-df9b-4521-9e96-b3d650221c9b) |

