# Train ABINet
# v0.8
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/train_abinet.yaml
# v0.9
CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/train_abinet_wo_iter.yaml

# Pre-train Vision model
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model.yaml

# Pre-train Language model
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_language_model.yaml
