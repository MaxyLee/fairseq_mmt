#!/bin/bash
#SBATCH --job-name              selattn-train
#SBATCH --partition             gpu-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   70G
#SBATCH --gres                  gpu:1
#SBATCH --output                /data/home/yc27434/projects/mmt/logs/selattn-train.%j.out
#SBATCH --error                 /data/home/yc27434/projects/mmt/logs/selattn-train.%j.err
#SBATCH --mail-type		NONE
#SBATCH --mail-user		yc27434@connect.um.edu.mo

source /etc/profile
source /etc/profile.d/modules.sh

#Adding modules
# module add cuda/9.2.148
# module add amber/18/gcc/4.8.5/cuda/9

ulimit -s unlimited

#Your program starts here 
set -e

device=0
task=a30k-en2zh
# task=flickr30k-en2zh
mask_data=mask0
tag=$mask_data
save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

if [ $task == 'multi30k-en2de' ]; then
	src_lang=en
	tgt_lang=de
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-de
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-de.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-de.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-de.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-de.mask4
        elif [ $mask_data == "maskc" ]; then
	        data_dir=multi30k.en-de.maskc
        elif [ $mask_data == "maskp" ]; then
	        data_dir=multi30k.en-de.maskp
	fi
elif [ $task == 'multi30k-en2fr' ]; then
	src_lang=en
	tgt_lang=fr
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-fr
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-fr.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-fr.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-fr.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-fr.mask4
        elif [ $mask_data == "maskc" ]; then
	        data_dir=multi30k.en-fr.maskc
        elif [ $mask_data == "maskp" ]; then
	        data_dir=multi30k.en-fr.maskp
	fi
elif [ $task == 'multi30k-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-zh
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-zh.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-zh.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-zh.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-zh.mask4
        elif [ $mask_data == "maskc" ]; then
	        data_dir=multi30k.en-zh.maskc
        elif [ $mask_data == "maskp" ]; then
	        data_dir=multi30k.en-zh.maskp
	fi
elif [ $task == 'flickr30k-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=flickr30k.en-zh
elif [ $task == 'a30k-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=a30k_1st.en-zh
fi


criterion=label_smoothed_cross_entropy
fp16=1 #0
lr=0.005
warmup=2000
max_tokens=4096
update_freq=1
keep_last_epochs=10
patience=10
max_update=8000
dropout=0.3

arch=transformer_tiny

# multi-feature
#image_feat_path=data/vit_large_patch16_384 data/vit_tiny_patch16_384
#image_feat_dim=1024 192

cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update
  --find-unused-parameters
  --patience $patience
  --keep-last-epochs $keep_last_epochs"


if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

export CUDA_VISIBLE_DEVICES=$device
# cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
# tail -f $save_dir/train.log
