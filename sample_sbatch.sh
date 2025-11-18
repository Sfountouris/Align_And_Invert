#!/bin/bash
#
#SBATCH --job-name=example-gpu # Job name for tracking
#SBATCH --partition=wmlg-ada # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=0-22:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

python3  sample_condition.py \
  --model SiT-XL/2 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --mode=ode \
  --num-steps=1000 \
  --cfg-scale=1 \
  --guidance-high=0.7 \
  --task_config=configs/super_resolution_config.yaml
