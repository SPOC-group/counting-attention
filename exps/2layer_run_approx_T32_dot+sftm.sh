#!/bin/bash
for i in {1..5}; do
    for p in 1 3 6 11 23 45 91 2 4 8 16 32 64 128; do
        for d in 1 3 6 11 23 45 91 2 4 8 16 32 64 128; do
            python3 memory_exp_2layer.py --seq_len=10 --T=32 --num_samples=10000 --p=$p --model_dim=$d --n_epochs=1000 --lr=0.001 --project_name=T32_L32_2layer --include_validation  --attention_input=only_sem  --dataset_type=backward
        done
    done
done
