#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
nohup tensorboard --logdir "_log" &
python -u start.py --run train