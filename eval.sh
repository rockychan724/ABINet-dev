export http_proxy=http://222.199.197.95:7890 https_proxy=http://222.199.197.95:7890
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config configs/pretrain_vision_model_v2.0.yaml \
    --phase test \
    --checkpoint workdir/pretrain-vision-model-v2.0/best-pretrain-vision-model-v2.0.pth \
    --model_eval vision
