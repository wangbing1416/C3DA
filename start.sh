#!/bin/bash
#SBATCH -J C3DA
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH -p gpu
#SBATCH -o out_scream.out

python train.py --dataset restaurant --prompt_name lora --ft_epoch 104 --seed 1000 --model_name roberta
python train.py --dataset restaurant --prompt_name lora --ft_epoch 104 --seed 2000 --model_name roberta
python train.py --dataset restaurant --prompt_name lora --ft_epoch 104 --seed 3000 --model_name roberta
python train.py --dataset restaurant --prompt_name lora --ft_epoch 104 --seed 4000 --model_name roberta
python train.py --dataset restaurant --prompt_name lora --ft_epoch 104 --seed 5000 --model_name roberta