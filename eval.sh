CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config configs/train_abinet.yaml \
    --phase test \
    --checkpoint workdir_origin/train-abinet/best-train-abinet.pth
