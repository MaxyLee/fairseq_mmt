multimodal machine translation(MMT) 
# Our dependency

* PyTorch version == 1.9.1
* Python version == 3.6.7
* timm version == 0.4.12
* vizseq version == 0.1.15
* nltk verison == 3.6.4
* sacrebleu version == 1.5.1

# Requirements and Installation

* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.


# multi30k data & flickr30k entities
Multi30k data from [here](https://github.com/multi30k/dataset) and [here](https://www.statmt.org/wmt17/multimodal-task.html)  
flickr30k entities data from [here](https://github.com/BryanPlummer/flickr30k_entities)  
We get multi30k text data from [Revisit-MMT](https://github.com/LividWo/Revisit-MMT)
```bash
# create a directory
flickr30k
├─ flickr30k-images
├─ test2017-images
├─ test_2016_flickr.txt
├─ test_2017_flickr.txt
├─ test_2017_mscoco.txt
├─ test_2018_flickr.txt
├─ testcoco-images
├─ train.txt
└─ val.txt
```

# Image feature
```bash
# please read scripts/README.md
python3 scripts/get_img_feat.py --dataset train
```

# train and test
```bash
sh preprocess.sh
sh train_mmt.sh
sh translation_mmt.sh
```

# masking data
```bash
pip3 install stanfordcorenlp 
wget https://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip
cd fairseq_mmt
python3 record_masking_position.py 

cd data/masking
cd en2de
python3 match_origin2bpe_position.py
python3 get_bpe_position.py         # create mask1-4 data
python3 create_masking_multi30k.py  # create mask color&people data

sh preprocess_mmt.sh
```

# visualization
```bash
```
