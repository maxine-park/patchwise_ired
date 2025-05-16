#!/bin/bash
#SBATCH --job-name=inverse_a
#SBATCH --output=cluster_logs/inverse_a_%j.out
#SBATCH --error=cluster_logs/inverse_a_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

source ~/.bashrc
mamba activate myenv

mkdir -p logs cluster_logs

python train.py \
    --model mlp \
    --dataset inverse \
    --rank 4 \
    --diffusion_steps 15 \
    --use-innerloop-opt True \
    --supervise-energy-landscape True \
    --patchwise_inference False \
    --data-workers 4 \
    > logs/inverse_a_mlp.log 2>&1
