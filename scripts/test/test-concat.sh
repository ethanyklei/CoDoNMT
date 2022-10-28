#/usr/bin/env bash

set -u 
set -eo pipefail

dataset=$1
cp_name=$2
export CUDA_VISIBLE_DEVICES=$3

src=en
tgt=ru

BIN_PATH=/path/to/bin/data
CP_DIR=/path/to/checkpoint

TOKENIZER=/home/yklei/nfs/toolkits/mosesdecoder/scripts/tokenizer/tokenizer.perl
LOWERCASES=/home/yklei/nfs/toolkits/mosesdecoder/scripts/tokenizer/lowercase.perl
MULTI_BLEU=/home/yklei/nfs/toolkits/mosesdecoder/scripts/generic/multi-bleu.perl

task=translation_doc_concat
beam=4
batch_size=32
max_len_a=1.2
max_len_b=10
lenpen=0.7

LOG_DIR=./log/test/${cp_name}
mkdir -p ${LOG_DIR}

LOG_PREFIX=${LOG_DIR}/generate-${use_ave}-${lenpen}-${max_len_a}-${max_len_b}


checkpoint_path=${CP_DIR}/${cp_name}/average.pt

if [ -e ${checkpoint_path} ]; then
echo `date`: ${checkpoint_path} exist.
else
python ./scripts/test/average_checkpoint.py \
        --inputs ${CP_DIR}/${cp_name} \
        --output  ${checkpoint_path} \
        --num-epoch-checkpoints 5
fi


cmd="python fairseq_generate ${BIN_PATH} \
        --path ${CP_DIR}/${cp_name}/average.pt \
        --source-lang ${src} --target-lang ${tgt} \
        --max-len-a ${max_len_a} --max-len-b ${max_len_b} \
        --lenpen ${lenpen} \
        --task ${task} \
        --beam ${beam} \
        --batch-size ${batch_size} \
        --remove-bpe
"

if [[ ${dataset} =~ "sent" ]]; then 
cmd=${cmd}" --sent"
fi 

cmd=${cmd}" > ${LOG_PREFIX}.results"

echo ${cmd}

eval ${cmd}

if [[ ${dataset} =~ "sent" ]]; then 

        grep -P "^T" ${LOG_PREFIX}.results | cut -f 2- | sed -e "s/ <\/s>//g" > ${LOG_PREFIX}.ref
        grep -P "^D" ${LOG_PREFIX}.results | cut -f 3- | sed -e "s/ <\/s>//g" > ${LOG_PREFIX}.sys

        sacrebleu ${LOG_PREFIX}.ref < ${LOG_PREFIX}.sys 

elif [[ ${dataset} =~ "doc" ]]; then 
        grep -P "^T" ${LOG_PREFIX}.results | cut -f 2- > ${LOG_PREFIX}.ref
        grep -P "^D" ${LOG_PREFIX}.results | cut -f 3- > ${LOG_PREFIX}.sys


        python ./scripts/test/doc_bleu.py \
                        --ref ${LOG_PREFIX}.ref \
                        --gen ${LOG_PREFIX}.sys \
                        --save-dir ${LOG_PREFIX} \
                        --fix-ctx 
        cat ${LOG_PREFIX}.sacrebleu
fi

