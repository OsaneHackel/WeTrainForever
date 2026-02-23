#!/bin/bash
#SBATCH --job-name=hockey
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-06:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anne.schlecht@student.uni-tuebingen.de

SEED=42
OPPONENT_TYPE="pool_with_sac" 
OPPONENT_ODDS='{"weak": 0.2, "strong": 0.2, "frozen_agent": 0.3, "sac": 0.3}'
# --opponent_odds "$OPPONENT_ODDS" \
# "weak", "current_self", "pretrained_self", "strong", "pool_basic_and_frozen_self"
MAX_EPISODES=12000
TRAIN_ITER=1
NOISE_TYPE="OrnsteinU" # OrnsteinU, Gaussian, Pink

# SAC opponent
SAC_PATH=~/BestAgents/SAC_best.pth

# get the agent from which to continue training on
RESUME_FROM_SAVED_PATH=~/outputs/1988388/saved/td3_final.pt
# 
SAVED_AGENT_PATH=~/outputs/1986284/saved/td3_final.pt
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
    --sac_path "$SAC_PATH" \
    --train_iter "$TRAIN_ITER" \
    --noise_type "$NOISE_TYPE" \
    --max_episodes "$MAX_EPISODES" \
    --output_dir "$RUN_DIR" \
    --seed "$SEED"

# save logs in my home directory
mkdir -p "$HOME/outputs"
cp -r "$RUN_DIR" "$HOME/outputs/$SLURM_JOB_ID"