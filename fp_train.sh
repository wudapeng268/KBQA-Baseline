#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
nohup tensorboard --logdir "_fp_log" &
python -u start_fp.py --run train