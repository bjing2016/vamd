```
CUDA_VISIBLE_DEVICES=0 python run_md.py --abs_pos_emb 37 --sample_dir /tmp/samples1 --run_name test_run --num_samples 1000 --num_steps 10000 --wandb & disown
CUDA_VISIBLE_DEVICES=1 python train.py --abs_pos_emb 37 --sample_dir /tmp/samples1 --run_name test_run --num_samples 1000 --val_batches 10000 --val_freq 10000 --ckpt_freq 100 --print_freq 100 --wandb & disown
# 1 minute of val per 10 minutes
```

Variables to adjust

-num_samples
-ckpt freq
-md length