checkpoints_path=$1
gpu=${2:-0}
subset=${3:-"test"}

mkdir -p $checkpoints_path/exp

CUDA_VISIBLE_DEVICES=$gpu python generate.py data/binary/wmt16_en_de_bpe32k  \
        --path "$checkpoints_path/checkpoint_best.pt" --gen-subset $subset \
        --beam 4 --batch-size 128 --remove-bpe  --lenpen 0.6 > $checkpoints_path/exp/${subset}_gen.out 

GEN=$checkpoints_path/exp/${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
python score.py --sys $SYS --ref $REF | tee $checkpoints_path/exp/checkpoint_best.result
