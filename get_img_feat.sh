#!/bin/bash
#SBATCH --job-name              get-img-feat
#SBATCH --partition             localQ
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   100G
#SBATCH --gres                  gpu:1
#SBATCH --output                /home/user/yc27434/projects/mmt/logs/get-img-feat.%j.out
#SBATCH --error                 /home/user/yc27434/projects/mmt/logs/get-img-feat.%j.err
#SBATCH --mail-type		NONE
#SBATCH --mail-user		yc27434@connect.um.edu.mo

#Your program starts here 
# F30K_PATH=/data/home/yc27434/projects/mmt/data/multi30k-dataset/data/task1/flickr30k-images
# python scripts/get_img_feat.py --dataset train --model vit_base_patch16_384 --path $F30K_PATH

CUDA_VIDIBLE_DEVICES=0 python scripts/get_img_feat.py