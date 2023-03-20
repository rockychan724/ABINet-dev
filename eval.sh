
# It works when testing the models of v2.x version
export http_proxy=http://222.199.197.95:7890 https_proxy=http://222.199.197.95:7890

# test vision model
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config configs/pretrain_vision_model_v2.1.yaml \
    --phase test \
    --checkpoint workdir/pretrain-vision-model-v2.1/best-pretrain-vision-model-v2.1.pth \
    --model_eval vision

# test language model
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config configs/pretrain_language_model_v0.7.yaml \
#     --phase test \
#     --checkpoint workdir/pretrain-language-model-v0.7/best-pretrain-language-model.pth \
#     --model_eval language

# test the whole model
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config configs/train_abinet_v1.1.yaml \
#     --phase test \
#     --checkpoint workdir/train-abinet-v1.1/best-train-abinet-v1.1.pth \
