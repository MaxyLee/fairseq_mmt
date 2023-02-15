src='en'
tgt='de'

# TEXT=data/multi30k-en-$tgt
TEXT=data/WIT-$src-$tgt
DICT=data-bin/multi30k.en-$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --srcdict $DICT/dict.en.txt \
  --validpref $TEXT/val \
  --destdir data-bin/WIT.en-$tgt \
  --workers 8 --joined-dictionary 
  # --trainpref $TEXT/train \
  # --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \