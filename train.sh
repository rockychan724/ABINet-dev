# Train ABINet
# v0.8
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/train_abinet.yaml
# v0.9
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/train_abinet_wo_iter.yaml
# v1.1
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/train_abinet_v1.1.yaml

# Pre-train Vision model
# v0.5
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model_v0.5.yaml
# v1.0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model_v1.0.yaml
# v1.2
CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model_v1.2.yaml

# Pre-train Language model
# v0.7
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_language_model.yaml
