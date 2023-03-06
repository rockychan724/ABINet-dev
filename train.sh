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
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model_v1.2.yaml
# v1.3
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model_v1.3.yaml
# v1.4
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model_v1.4.yaml

# v2.x 系列的模型涉及 gpt-3 和 bert 预训练语言模型，需要构建数据集初始化时需要访问网络，
# 使用代理，初始化的速度更快
export http_proxy=http://222.199.197.95:7890 https_proxy=http://222.199.197.95:7890
# v2.0
CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_vision_model_v2.0.yaml

# Pre-train Language model
# v0.7
# CUDA_VISIBLE_DEVICES=0,1 python main.py --config=configs/pretrain_language_model.yaml
