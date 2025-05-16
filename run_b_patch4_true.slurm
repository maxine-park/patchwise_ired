#!/bin/bash
#SBATCH --job-name=inverse_b
#SBATCH --output=cluster_logs/inverse_b_%j.out
#SBATCH --error=cluster_logs/inverse_b_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

source ~/.bashrc
mamba activate myenv

mkdir -p logs cluster_logs

python train.py \
    --model mlp-patch \
    --dataset inverse \
    --rank 4 \
    --diffusion_steps 15 \
    --patchsize 4 \
    --patchwise_inference True \
    --use-innerloop-opt True \
    --supervise-energy-landscape True \
    --data-workers 4 \
    > logs/inverse_b_patch4_true.log 2>&1
