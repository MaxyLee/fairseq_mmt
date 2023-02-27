#Your program starts here 
#!/usr/bin/bash
set -e

# set device
gpu=3
export CUDA_VISIBLE_DEVICES=$gpu

model_root_dir=checkpoints

M30K_DATA_PATH=/home/maxinyu/projects/mmt/data/multi30k-dataset/data/task1/raw
MSCTD_DATA_PATH=/home/maxinyu/projects/mmt/data/MSCTD/MSCTD_data/enzh
AM_DATA_PATH=/home/maxinyu/projects/mmt/data/3am
# set task
task=m30k_3am-en2zh
# task=m30k_test3am-en2zh
# task=msctd_test3am-en2zh
# task=3am-en2zh
mask_data=mask0

# who=test3	#test1, test2
# random_image_translation=0 #1
length_penalty=0.8

arch=transformer
arch=transformer_tiny
# set tag
# model_dir_tag=$mask_data
model_dir_tag=$arch-2

if [ $task == 'm30k_test3am-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=m30k_test3am.en-zh
elif [ $task == 'msctd_test3am-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=msctd_test3am.en-zh
elif [ $task == '3am-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=3am.en-zh
elif [ $task == 'm30k_3am-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=m30k_3am.en-zh
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

for who in test test1 test2;
do
	output=$model_dir/translation_$who.log

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

	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test' ]; then
		ref=$M30K_DATA_PATH/test_2016_flickr.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test1' ]; then
		ref=$M30K_DATA_PATH/test_2017_flickr.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test2' ]; then
		ref=data/multi30k/test.coco.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test3' ]; then
		ref=$AMBIG_DATA_PATH/test.zh
	elif [ $task == "m30k_ambig1-en2zh" ] && [ $who == 'test4' ]; then
		ref=$AM_DATA_PATH/test.zh
	elif [ $task == "m30k_test3am-en2zh" ] && [ $who == 'test' ]; then
		ref=$AM_DATA_PATH/test.zh
	elif [ $task == "msctd_ambig1-en2zh" ] && [ $who == 'test' ]; then
		ref=$MSCTD_DATA_PATH/test.zh
	elif [ $task == "msctd_ambig1-en2zh" ] && [ $who == 'test1' ]; then
		ref=$AMBIG_DATA_PATH/test.zh
	elif [ $task == "msctd_ambig1-en2zh" ] && [ $who == 'test2' ]; then
		ref=$AM_DATA_PATH/test.zh
	elif [ $task == "msctd_test3am-en2zh" ] && [ $who == 'test' ]; then
		ref=$AM_DATA_PATH/test.zh
	elif [ $task == "3am-en2zh" ] && [ $who == 'test' ]; then
		ref=$AM_DATA_PATH/test.zh
	elif [ $task == "m30k_3am-en2zh" ] && [ $who == 'test' ]; then
		ref=$M30K_DATA_PATH/test_2016_flickr.zh
	elif [ $task == "m30k_3am-en2zh" ] && [ $who == 'test1' ]; then
		ref=$M30K_DATA_PATH/test_2017_flickr.zh
	elif [ $task == "m30k_3am-en2zh" ] && [ $who == 'test2' ]; then
		ref=$AM_DATA_PATH/test.zh
	fi

	source /home/maxinyu/env/miniconda3/etc/profile.d/conda.sh
	conda activate base
	hypo=$model_dir/hypo-$who.sorted
	python eval.py $hypo $ref > $model_dir/eval-$who.log
	cat $model_dir/eval-$who.log
done

# cal gate, follow Revisit-MMT
#python3 scripts/visual_awareness.py --input $model_dir_tag/gated.txt 

# cal accurary
# python3 cal_acc.py $hypo $who $task
