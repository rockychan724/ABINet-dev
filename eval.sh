CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/train_abinet_wo_iter.yaml \
    --phase test \
    --checkpoint workdir/train-abinet-wo-iter-v0.9/best-train-abinet-wo-iter-v0.9.pth \
    # --model_eval language
