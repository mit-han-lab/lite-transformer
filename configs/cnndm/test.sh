  
checkpoint_path=$1
output_path=$checkpoint_path/exp
gpu=${2:-0}
dataset=${3:-"test"}
mkdir -p $output_path

CUDA_VISIBLE_DEVICES=$gpu python generate.py data/binary/cnndm --path "$checkpoint_path/checkpoint_best.pt" --remove-bpe --gen-subset $dataset \
  --batch-size 6 --min-len 55 --max-len-b 140  --beam 4 --lenpen 2.0 --no-repeat-ngram-size 3 > $output_path/cnn_dailymail.out

GEN=$output_path/cnn_dailymail.out
SYS=$GEN.sys
REF=$GEN.ref
grep ^H $GEN | cut -f3- > $SYS
grep ^T $GEN | cut -f2- > $REF

export CLASSPATH=`pwd`/configs/cnndm/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

cat $SYS | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $SYS.tokenized
cat $REF | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $REF.target

files2rouge $SYS.tokenized $REF.target | tee $output_path/rouge.result
