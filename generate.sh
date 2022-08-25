#!/bin/bash
#SBATCH -J C3DA
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH -p gpu
#SBATCH -o out_scream.out

python generate.py --num_epoch 100 --prompt_name lora