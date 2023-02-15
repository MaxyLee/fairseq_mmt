#!/bin/bash
#SBATCH --job-name              get-img-feat
#SBATCH --partition             gpu-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   150G
#SBATCH --gres                  gpu:1
#SBATCH --output                /data/home/yc27434/projects/mmt/logs/get-img-feat.%j.out
#SBATCH --error                 /data/home/yc27434/projects/mmt/logs/get-img-feat.%j.err
#SBATCH --mail-type		NONE
#SBATCH --mail-user		yc27434@connect.um.edu.mo

source /etc/profile
source /etc/profile.d/modules.sh

#Adding modules
# module add cuda/9.2.148
# module add amber/18/gcc/4.8.5/cuda/9

ulimit -s unlimited

#Your program starts here 
# F30K_PATH=/data/home/yc27434/projects/mmt/data/multi30k-dataset/data/task1/flickr30k-images
# python scripts/get_img_feat.py --dataset train --model vit_base_patch16_384 --path $F30K_PATH

python scripts/get_img_feat.py