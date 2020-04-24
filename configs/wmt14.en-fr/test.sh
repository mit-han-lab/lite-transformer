checkpoints_path=$1
gpu=${2:-0}
subset=${3:-"test"}

mkdir -p $checkpoints_path/exp

CUDA_VISIBLE_DEVICES=$gpu python generate.py data/binary/wmt14_en_fr  \
        --path "$checkpoints_path/checkpoint_best.pt" --gen-subset $subset \
        --beam 4 --batch-size 128 --remove-bpe  --lenpen 0.6 > $checkpoints_path/exp/${subset}_gen.out 

GEN=$checkpoints_path/exp/${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3- > $SYS
grep ^T $GEN | cut -f2- > $REF
python score.py --sys $SYS --ref $REF | tee $checkpoints_path/exp/checkpoint_best.result
