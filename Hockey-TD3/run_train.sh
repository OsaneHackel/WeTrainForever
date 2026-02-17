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

SEED=42
OPPONENT_TYPE="current_self"
MAX_EPISODES=50000
TRAIN_ITER=1
NOISE_TYPE="Pink" # OrnsteinU, Gaussian, Pink

# get the agent from which to continue training on
RESUME_FROM_SAVED_PATH=~/outputs/1985807/saved/td3_final.pt

# set up a run directory for this job to write training logs
RUN_DIR=/scratch/$SLURM_JOB_ID/hockey_td3
mkdir -p "$RUN_DIR"

# path to your singularity image
SIMG=~/singularity_build/hockey_td3.simg

# activate your environment
# source activate RL_project   # or: conda activate RL_project
# run training
## python main.py train --opponent_type weak --max_episodes 50000

singularity exec --nv "$SIMG" python3 ~/Hockey-TD3/main.py train \
    --opponent_type "$OPPONENT_TYPE" \
    --resume_from_saved_path "$RESUME_FROM_SAVED_PATH" \
    --train_iter "$TRAIN_ITER" \
    --noise_type "$NOISE_TYPE" \
    --max_episodes "$MAX_EPISODES" \
    --output_dir "$RUN_DIR" \
    --seed "$SEED"

# save logs in my home directory
mkdir -p "$HOME/outputs"
cp -r "$RUN_DIR" "$HOME/outputs/$SLURM_JOB_ID"