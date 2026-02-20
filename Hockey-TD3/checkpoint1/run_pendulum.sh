#!/bin/bash
#SBATCH --job-name=hockey
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anne.schlecht@student.uni-tuebingen.de

NOISE_TYPE="Gaussian"   # Gaussian, OrnsteinU, Pink
EXPLORATION_NOISE=0.1
MAX_EPISODES=2000
TRAIN_ITER=2
SEED=0

# output directory for saved stats/checkpoints
RUN_DIR=/scratch/$SLURM_JOB_ID/pendulum_td3
mkdir -p "$RUN_DIR"
trap 'mkdir -p "$HOME/outputs" && cp -r "$RUN_DIR" "$HOME/outputs/$SLURM_JOB_ID"' EXIT

SIMG=~/singularity_build/hockey_td3.simg

singularity exec --nv "$SIMG" python3 ~/Hockey-TD3/checkpoint1/train_pendulum.py \
    --noise_type "$NOISE_TYPE" \
    --exploration_noise "$EXPLORATION_NOISE" \
    --max_episodes "$MAX_EPISODES" \
    --train_iter "$TRAIN_ITER" \
    --seed "$SEED"