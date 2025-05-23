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
source ~/patchwise_ired/myenv/bin/activate

mkdir -p logs cluster_logs

python train.py \
    --model mlp-patch \
    --dataset inverse \
    --rank 4 \
    --diffusion_steps 15 \
    --patchsize 8 \
    --patchwise_inference False \
    --use-innerloop-opt True \
    --supervise-energy-landscape True \
    --data-workers 4 \
    > logs/inverse_b_patch8_false.log 2>&1
