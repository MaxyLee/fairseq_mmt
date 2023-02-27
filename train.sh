set -e

device=2
# task=m30k_3am-en2zh
# task=m30k_test3am-en2zh
task=msctd_test3am-en2zh
# task=3am-en2zh
# task=flickr30k-en2zh
mask_data=mask0
tag=$mask_data

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


criterion=label_smoothed_cross_entropy
fp16=1 #0
lr=0.005
warmup=16000
max_tokens=4096
update_freq=1
keep_last_epochs=10
patience=10
max_update=64000
dropout=0.1

arch=transformer
# arch=transformer_tiny

save_dir=checkpoints/$task/$arch-2
if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

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
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
# tail -f $save_dir/train.log
