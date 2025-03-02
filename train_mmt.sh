#Your program starts here 
set -e

device=2
# task=multi30k-en2zh
task=m30k_test3am-en2zh
# task=msctd_ambig1-en2zh
# task=3am-en2zh
image_feat=vit_base_patch16_384
mask_data=mask0
tag=$image_feat/$mask_data

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
	if [ $mask_data == "mask0" ]; then
		data_dir=flickr30k.en-zh
	fi
elif [ $task == 'm30k_test3am-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=m30k_test3am.en-zh
	image_feat_root=data/m30k_test3am
elif [ $task == 'msctd_ambig1-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=msctd_ambig1.en-zh
elif [ $task == '3am-en2zh' ]; then
	src_lang=en
	tgt_lang=zh
	data_dir=3am.en-zh
fi
criterion=label_smoothed_cross_entropy
fp16=1 #0
lr=0.0075
warmup=8000
max_tokens=4096
update_freq=1
keep_last_epochs=10
patience=10
max_update=32000
dropout=0.3

arch=image_multimodal_transformer_SA_top
# arch=image_multimodal_transformer_SA_top_base
SA_attention_dropout=0.1
SA_image_dropout=0.1

save_dir=checkpoints/$task/$arch

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
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
  --task image_mmt --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update
  --find-unused-parameters
  --patience $patience
  --keep-last-epochs $keep_last_epochs"

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

if [ -n "$SA_image_dropout" ]; then
cmd=${cmd}" --SA-image-dropout "${SA_image_dropout}
fi
if [ -n "$SA_attention_dropout" ]; then
cmd=${cmd}" --SA-attention-dropout "${SA_attention_dropout}
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
# tail -f $save_dir/train.log
