#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py --dataset web-iNat --base_lr 3e-3 --batch_size 160 --epoch 60 --drop_rate 0.3 --T_k 10 --weight_decay 1e-5 --n_classes 5089 --net resnet50 --step 0
