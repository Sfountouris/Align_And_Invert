# Align and Invert Sampling

We present Align and Invert, a method for solving inverse problems by enforcing representation alignment between diffusion models and pretrained self-supervised visual encoders such as DINOv2.

<p align="center">
  <img src="/dcs/pg24/u5671205/REPA/Aligned_Dps/Align_Invert/data/diagram.jpg" width="500">
</p>

## Installation

First, create the environment and install all dependencies:

```bash
conda env create -f environment.yml
```

clone external repositories:

```bash
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
git clone https://github.com/LeviBorodenko/motionblur motionblur
```

## Running the Code

## Running the Code

To run the main sampling script, use:

```bash
python3 sample_condition.py \
  --model SiT-XL/2 \
  --l_repa=0.01 \
  --learning_rate=2 \
  --num-steps=1000 \
  --task_config=configs/super_resolution_config.yaml
```
---------------------------------------------------------------
Suggested Hyperparameters for Each Inverse Problem

| Task               | Config file                               | l_repa  | learning_rate |
|--------------------|--------------------------------------------|--------|----------------|
| Super-resolution   | configs/super_resolution_config.yaml       | 0.01   |      2         |
| Gaussian Deblurring| configs/gaussian_deblur_config.yaml        | 0.05   |      0.25      |
| Motion Deblurring  | configs/motion_deblur_config.yaml          | 0.01   |      0.5       |
| Box Inpainting     | configs/inpainting_config.yaml             | 0.01   |      0.5       |

---------------------------------------------------------------

## Credits

Our code was based on the implementations provided in the following repositories:

- https://github.com/DPS2022/diffusion-posterior-sampling  
- https://github.com/sihyun-yu/REPA.git  
- https://github.com/soominkwon/resample.git


