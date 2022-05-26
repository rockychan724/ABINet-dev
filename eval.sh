CUDA_VISIBLE_DEVICES=2 python main.py \
    --config configs/train_abinet.yaml \
    --phase test \
    --checkpoint workdir_origin/train-abinet/best-train-abinet.pth
