checkpoints_path=$1
gpu=${2:-0}
subset=${3:-"test"}
avg_checkpoints=${4:-10}
model=average_model_$avg_checkpoints.pt
output_path=$checkpoints_path

mkdir -p $output_path/exp

python scripts/average_checkpoints.py --inputs $output_path \
   --num-epoch-checkpoints $avg_checkpoints --output $output_path/$model

CUDA_VISIBLE_DEVICES=$gpu python ./generate.py data/binary/iwslt14.tokenized.de-en \
    --path $output_path/$model --gen-subset $subset \
    --batch-size 128 --beam 4 --remove-bpe > $output_path/exp/${subset}_gen.out 

GEN=$output_path/exp/${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3-  > $SYS
grep ^T $GEN | cut -f2-  > $REF
python score.py --sys $SYS --ref $REF | tee $checkpoints_path/exp/avg_checkpoints.result
