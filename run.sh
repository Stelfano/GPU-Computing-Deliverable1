#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --job-name=spvm
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err

module load CUDA/11.8.0

./bin/spvm $1
