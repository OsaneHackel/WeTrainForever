#!/bin/bash
#SBATCH --job-name=hockey
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anne.schlecht@student.uni-tuebingen.de

SEED=42
OPPONENT_TYPE="pool_and_self_play" 
OPPONENT_ODDS='{"weak": 0.15, "strong": 0.25, "sac": 0.1, "current_self": 0.4, "frozen_agent": 0.1}'
# --opponent_odds "$OPPONENT_ODDS" \
# "weak", "current_self", "pretrained_self", "strong", "pool_basic_and_frozen_self"
MAX_EPISODES=15000
TRAIN_ITER=1
NOISE_TYPE="Pink" # OrnsteinU, Gaussian, Pink

# SAC opponent
SAC_PATH=~/BestAgents/SAC_best_3.pth

# get the agent from which to continue training on
RESUME_FROM_SAVED_PATH=~/outputs/1989131/saved/td3_final.pt
# 
SAVED_AGENT_PATH=~/outputs/1989131/saved/td3_final.pt
# 1986341 strong, Iter=1, Pink, Prioritized ER

# set up a run directory for this job to write training logs
RUN_DIR=/scratch/$SLURM_JOB_ID/hockey_td3
mkdir -p "$RUN_DIR"
trap 'mkdir -p "$HOME/outputs" && cp -r "$RUN_DIR" "$HOME/outputs/$SLURM_JOB_ID"' EXIT

# path to your singularity image
SIMG=~/singularity_build/hockey_td3.simg

# activate your environment
# source activate RL_project   # or: conda activate RL_project
# run training
## python main.py train --opponent_type weak --max_episodes 50000


# --use_PrioritizedExpReplay \

singularity exec --nv "$SIMG" python3 ~/Hockey-TD3/main.py train \
    --opponent_type "$OPPONENT_TYPE" \
    --opponent_odds "$OPPONENT_ODDS" \
    --saved_agent_path "$SAVED_AGENT_PATH" \
    --resume_from_saved_path "$RESUME_FROM_SAVED_PATH" \
    --use_PrioritizedExpReplay \
    --alternate_sides \
    --clip_noise=0.2 \
    --sac_path "$SAC_PATH" \
    --train_iter "$TRAIN_ITER" \
    --noise_type "$NOISE_TYPE" \
    --max_episodes "$MAX_EPISODES" \
    --output_dir "$RUN_DIR" \
    --seed "$SEED"

# save logs in my home directory
mkdir -p "$HOME/outputs"
cp -r "$RUN_DIR" "$HOME/outputs/$SLURM_JOB_ID"