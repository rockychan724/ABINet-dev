#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0
tests=(IC13_857 SVT IIIT5k_3000 IC15_1811 SVTP CUTE80)
data_base=data/evaluation/
for i in "${tests[@]}"; do
  echo $i
  python main.py --config=configs/pretrain_vision_model_v0.5.yaml \
    --test_root=$data_base$i \
    --phase=test \
    --checkpoint=workdir/pretrain-vision-model-v0.5/best-pretrain-vision-model.pth \
    --model_eval=vision
done
