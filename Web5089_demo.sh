#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python demo.py --data web-iNat --model model/best_epoch.pth --n_classes 5089 --net resnet50
