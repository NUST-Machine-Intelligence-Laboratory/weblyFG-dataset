#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export DATA='web-aircraft'
export N_CLASSES=100

python main.py --dataset ${DATA} --base_lr 1e-3 --batch_size 64 --epoch 200 --drop_rate 0.35 --T_k 10 --weight_decay 1e-8 --n_classes ${N_CLASSES} --net bcnn --step 1

sleep 300

python main.py --dataset ${DATA} --base_lr 1e-4 --batch_size 32 --epoch 200 --drop_rate 0.35 --T_k 10 --weight_decay 1e-5 --n_classes ${N_CLASSES} --net bcnn --step 2
