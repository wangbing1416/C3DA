#!/bin/bash
#SBATCH -J C3DA
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH -p gpu
#SBATCH -o out_scream.out

#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --seed 1000 --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --seed 2000 --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --seed 3000 --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --seed 4000 --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --seed 5000 --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.1 --margin 0.5 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.1 --margin 0.5 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.1 --margin 0.5 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.1 --margin 0.5 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.1 --margin 0.5 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.3 --margin 0.5 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.3 --margin 0.5 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.3 --margin 0.5 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.3 --margin 0.5 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.3 --margin 0.5 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.5 --margin 0.5 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.5 --margin 0.5 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.5 --margin 0.5 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.5 --margin 0.5 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 0.5 --margin 0.5 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 2.0 --margin 0.5 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 2.0 --margin 0.5 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 2.0 --margin 0.5 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 2.0 --margin 0.5 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 2.0 --margin 0.5 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.1 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.1 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.1 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.1 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.1 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.2 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.2 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.2 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.2 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.2 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.3 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.3 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.3 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.3 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.3 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.4 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.4 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.4 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.4 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.4 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.6 --cl_loss_fac 2.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.6 --cl_loss_fac 2.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.6 --cl_loss_fac 2.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.6 --cl_loss_fac 2.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.6 --cl_loss_fac 2.0 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 0.5 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 0.5 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 0.5 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 0.5 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 0.5 --seed 5000 --withAugment --withCL --model_name roberta
#
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 1.0 --seed 1000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 1.0 --seed 2000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 1.0 --seed 3000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 1.0 --seed 4000 --withAugment --withCL --model_name roberta
#python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 1.0 --seed 5000 --withAugment --withCL --model_name roberta

python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 5.0 --seed 1000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 5.0 --seed 2000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 5.0 --seed 3000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 5.0 --seed 4000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 5.0 --seed 5000 --withAugment --withCL --model_name roberta

python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 10.0 --seed 1000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 10.0 --seed 2000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 10.0 --seed 3000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 10.0 --seed 4000 --withAugment --withCL --model_name roberta
python train.py --dataset twitter --prompt_name lora --ft_epoch 103 --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 10.0 --seed 5000 --withAugment --withCL --model_name roberta