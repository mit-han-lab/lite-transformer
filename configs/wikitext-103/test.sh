checkpoint_path=$1
gpu=${2:-0}
subset=${3:-test}

mkdir -p $checkpoint_path/exp

CUDA_VISIBLE_DEVICES=$gpu python eval_lm.py \
    data/binary/wikitext-103 \
    --path "$checkpoint_path/checkpoint_best.pt" \
    --sample-break-mode complete \
    --max-tokens 3072 \
    --context-window 2560 \
    --softmax-batch 1024 \
    --gen-subset $subset | tee "$checkpoint_path/exp/checkpoint_best.$subset.result"