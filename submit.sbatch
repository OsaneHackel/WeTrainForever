#!/bin/bash

#SBATCH --job-name=SLS-2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu-bind=none
#SBATCH --time=3-00:00:00
#SBATCH -p 2080-galvani

srun --gres-flags=allow-task-sharing uv run -m wtf.train-2