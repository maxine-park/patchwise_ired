#!/bin/bash
#SBATCH --job-name=inverse_experiments
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Activate mamba environment
source ~/.bashrc
mamba activate iredenv

mkdir -p logs cluster_logs

COMMON_ARGS="--dataset inverse --rank 4 --diffusion_steps 15 --use-innerloop-opt True --supervise-energy-landscape True"

# a) Normal EBM
nohup python train.py \
    --model mlp \
    --patchwise_inference False \
    $COMMON_ARGS \
    > logs/inverse_a_mlp.log 2>&1 &

# b) Patch 4, inference True
nohup python train.py \
    --model mlp-patch \
    --patchsize 4 \
    --patchwise_inference True \
    $COMMON_ARGS \
    > logs/inverse_b_patch4_true.log 2>&1 &

# c) Patch 4, inference False
nohup python train.py \
    --model mlp-patch \
    --patchsize 4 \
    --patchwise_inference False \
    $COMMON_ARGS \
    > logs/inverse_c_patch4_false.log 2>&1 &

# d) Patch 8, inference True
nohup python train.py \
    --model mlp-patch \
    --patchsize 8 \
    --patchwise_inference True \
    $COMMON_ARGS \
    > logs/inverse_d_patch8_true.log 2>&1 &

# e) Patch 8, inference False
nohup python train.py \
    --model mlp-patch \
    --patchsize 8 \
    --patchwise_inference False \
    $COMMON_ARGS \
    > logs/inverse_e_patch8_false.log 2>&1 &

wait
