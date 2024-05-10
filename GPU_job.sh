#!/bin/bash

#SBATCH --job-name="Learning_simulation"
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --account= your_account

module load 2022r2
module load openmpi
module load miniconda3
module load py-numpy
module load py-torch
module load py-matplotlib

srun python ppo.py