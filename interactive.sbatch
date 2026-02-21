#!/bin/bash

#SBATCH --job-name=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu-bind=none
#SBATCH --time=1-00:00:00
#SBATCH -p 2080-galvani

sleep 100000000