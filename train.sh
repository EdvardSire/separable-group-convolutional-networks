#! /bin/bash
python3 run_experiment.py \
    --model ckgresnet \
    --group Sim2 \
    --num_group_elements 16,3 \
    --max_scale 1.74 \
    --sampling_method uniform \
    --hidden_sizes 32,32,64 \
    --dataset SUAS \
    --epochs 300 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --optim adam \
    --kernel_size 7 \
    --stride 1 \
    --dropout 0 \
    --weight_decay 1e-4 \
    --ck_net_num_hidden 1 \
    --ck_net_hidden_size 64 \
    --first_omega_0 10 \
    --omega_0 10 \
    --implementation separable+2d \
    --pooling 1 \
    --normalisation batchnorm \
    --learning_rate_cosine 1 \
    --padding 1 \
    --no_wandb
