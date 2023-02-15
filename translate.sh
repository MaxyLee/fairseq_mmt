#!/bin/bash
#SBATCH --job-name              selattn-eval
#SBATCH --partition             gpu-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   70G
#SBATCH --gres                  gpu:1
#SBATCH --output                /data/home/yc27434/projects/mmt/logs/selattn-eval.%j.out
#SBATCH --error                 /data/home/yc27434/projects/mmt/logs/selattn-eval.%j.err
#SBATCH --mail-type		NONE
#SBATCH --mail-user		yc27434@connect.um.edu.mo

source /etc/profile
source /etc/profile.d/modules.sh

#Adding modules
# module add cuda/9.2.148
# module add amber/18/gcc/4.8.5/cuda/9

ulimit -s unlimited

#Your program starts here 
#!/usr/bin/bash
set -e

# set device
gpu=0

model_root_dir=checkpoints

# set task
task=multi30k-en2zh
mask_data=mask0

who=test	#test1, test2
# random_image_translation=0 #1
length_penalty=0.8

# set tag
model_dir_tag=$mask_data

if [ $task == "multi30k-en2de" ]; then
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
fi


# data set
# ensemble=10
batch_size=128
beam=5
src_lang=en

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
	if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
		PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
	fi
	checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation_$who.log

export CUDA_VISIBLE_DEVICES=$gpu

cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang 
  --path $model_dir/$checkpoint 
  --gen-subset $who 
  --batch-size $batch_size --beam $beam --lenpen $length_penalty 
  --quiet --remove-bpe
  --output $model_dir/hypo-$who.txt" 


cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $model_dir/hypo-$who.txt $model_dir/hypo-$who.sorted

if [ $task == "multi30k-en2de" ] && [ $who == "test" ]; then
	ref=data/multi30k/test.2016.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test1" ]; then
	ref=data/multi30k/test.2017.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test2" ]; then
	ref=data/multi30k/test.coco.de

elif [ $task == "multi30k-en2fr" ] && [ $who == 'test' ]; then
	ref=data/multi30k/test.2016.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test1' ]; then
	ref=data/multi30k/test.2017.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test2' ]; then
	ref=data/multi30k/test.coco.fr

elif [ $task == "multi30k-en2zh" ] && [ $who == 'test' ]; then
	ref=data/multi30k/test.2016.zh
elif [ $task == "multi30k-en2zh" ] && [ $who == 'test1' ]; then
	ref=data/multi30k/test.2017.zh
elif [ $task == "multi30k-en2zh" ] && [ $who == 'test2' ]; then
	ref=data/multi30k/test.coco.zh
fi	

hypo=$model_dir/hypo-$who.sorted
python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log
cat $model_dir/meteor_$who.log

# cal gate, follow Revisit-MMT
#python3 scripts/visual_awareness.py --input $model_dir_tag/gated.txt 

# cal accurary
# python3 cal_acc.py $hypo $who $task
