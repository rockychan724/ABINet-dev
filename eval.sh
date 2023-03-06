CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config configs/pretrain_vision_model_v1.4.yaml \
    --phase test \
    --checkpoint workdir/pretrain-vision-model-v1.4/best-pretrain-vision-model-v1.4.pth \
    --model_eval vision
