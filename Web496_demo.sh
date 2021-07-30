#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export DATA='web-aircraft'
export N_CLASSES=100

python demo.py --data ${DATA} --model model/best_epoch.pth --n_classes ${N_CLASSES} --net bcnn
