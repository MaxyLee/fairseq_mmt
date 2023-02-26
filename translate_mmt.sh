#!/usr/bin/bash
set -e

# set device
gpu=0

M30K_DATA_PATH=/home/user/yc27434/projects/mmt/data/multi30k-dataset/data/task1/raw
MSCTD_DATA_PATH=/home/user/yc27434/projects/mmt/data/MSCTD/MSCTD_data/enzh
AMBIG_DATA_PATH=/home/user/yc27434/projects/mmt/code/VL-T5/datasets/mmt/1st
AMBIG_DATA_PATH2=/home/user/yc27434/projects/mmt/code/VL-T5/datasets/mmt/2nd
model_root_dir=checkpoints

# set task
# task=m30k_ambig1-en2zh
# task=msctd_ambig1-en2zh
task=3am-en2zh
mask_data=mask0
image_feat=vit_base_patch16_384


# who=test	#test1, test2
random_image_translation=0 #1
length_penalty=0.8

# arch=image_multimodal_transformer_SA_top
arch=image_multimodal_transformer_SA_top_base
# set tag
model_dir_tag=$arch-2

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
elif [ $task == 'm30k_ambig1-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=m30k_ambig1.en-zh
	image_feat_root=data
elif [ $task == 'msctd_ambig1-en2zh' ]; then
	image_feat=vit_tiny_patch16_384
	src_lang=en
	tgt_lang=zh
	data_dir=msctd_ambig1.en-zh
	image_feat_root=data/msctd
elif [ $task == '3am-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=3am.en-zh
	image_feat_root=data/3am
fi


if [ $image_feat == "vit_tiny_patch16_384" ]; then
	image_feat_path=$image_feat_root/$image_feat
	image_feat_dim=192
elif [ $image_feat == "vit_small_patch16_384" ]; then
	image_feat_path=$image_feat_root/$image_feat
	image_feat_dim=384
elif [ $image_feat == "vit_base_patch16_384" ]; then
	image_feat_path=$image_feat_root/$image_feat
	image_feat_dim=768
elif [ $image_feat == "vit_large_patch16_384" ]; then
	image_feat_path=$image_feat_root/$image_feat
	image_feat_dim=1024
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



export CUDA_VISIBLE_DEVICES=$gpu

for who in test ;
do
	output=$model_dir/translation_$who.log
	cmd="fairseq-generate data-bin/$data_dir 
	-s $src_lang -t $tgt_lang 
	--path $model_dir/$checkpoint 
	--gen-subset $who 
	--batch-size $batch_size --beam $beam --lenpen $length_penalty 
	--quiet --remove-bpe
	--task image_mmt
	--image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
	--output $model_dir/hypo-$who.txt" 

	if [ $random_image_translation -eq 1 ]; then
	cmd=${cmd}" --random-image-translation "
	fi

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

	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test' ]; then
		ref=$M30K_DATA_PATH/test_2016_flickr.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test1' ]; then
		ref=$M30K_DATA_PATH/test_2017_flickr.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test2' ]; then
		ref=data/multi30k/test.coco.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test3' ]; then
		ref=$AMBIG_DATA_PATH/test.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test4' ]; then
		ref=$AMBIG_DATA_PATH2/test.zh
	elif [ $task == "msctd_ambig1-en2zh" ] && [ $who == 'test' ]; then
		ref=$MSCTD_DATA_PATH/test.zh
	elif [ $task == "msctd_ambig1-en2zh" ] && [ $who == 'test1' ]; then
		ref=$AMBIG_DATA_PATH/test.zh
	elif [ $task == "msctd_ambig1-en2zh" ] && [ $who == 'test2' ]; then
		ref=$AMBIG_DATA_PATH2/test.zh
	elif [ $task == "3am-en2zh" ] && [ $who == 'test' ]; then
		ref=$AMBIG_DATA_PATH2/test.zh
	fi

	source /home/user/yc27434/env/miniconda3/etc/profile.d/conda.sh
	conda activate vlt5
	hypo=$model_dir/hypo-$who.sorted
	python eval.py $hypo $ref > $model_dir/eval-$who.log
	cat $model_dir/eval-$who.log
	conda deactivate
done
# cal gate, follow Revisit-MMT
#python3 scripts/visual_awareness.py --input $model_dir_tag/gated.txt 

# cal accurary
# python3 cal_acc.py $hypo $who $task
