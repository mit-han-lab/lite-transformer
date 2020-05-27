#!/bin/bash
# Please download the data from the google drive https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8
# Place the downloaded zip file into data/

DATA_DIR=data
TEXT=$DATA_DIR/wmt16_en_de_bpe32k
TAR=${1:-data/wmt16_en_de_bpe32k.tar.gz}
mkdir -p $TEXT
if [ "$TAR" == "" ]; then TAR=data/wmt16_en_de_bpe32k.tar.gz; fi
tar -xvzf $TAR -C "$DATA_DIR"

fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data/binary/wmt16_en_de_bpe32k \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary --workers 10
