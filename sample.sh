#!/bin/bash

python3  sample_condition.py \
  --model SiT-XL/2 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --mode=ode \
  --num-steps=25 \
  --cfg-scale=1 \
  --guidance-high=0.7 \
  --task_config=configs/super_resolution_config.yaml