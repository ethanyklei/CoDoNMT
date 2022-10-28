#/usr/bin/env bash

set -u 
set -eo pipefail

mode=$1
export CUDA_VISIBLE_DEVICES=$2

GPU_NUM=`echo "$2" | awk '{split($0,arr,",");print length(arr)}'`


BIN_PATH=/path/to/bin/data
SAVE_DIR=/path/to/save/model

LOG_DIR=./log
mkdir -p ${LOG_DIR}

src=en 
tgt=ru


# General
task=translation_doc_concat
seed=2021
lr_scheduler=inverse_sqrt
# Dataset
validate_interval=1
# Model
arch=transformer_concat_base
# Optimization
criterion=label_smoothed_cross_entropy
label_smoothing=0.1
warmup=4000
update_freq=1

if [ ${mode} == "sent" ]; then 

  echo `date`: training sent level model

  # specific args
  max_tokens=4096

  lr=5e-4

  warmup=4000

  patience=5

  dropout=0.1

  keep_last_epochs=5

  eval_bleu=1

  cp_name=${dataset}-${task}-${arch}-${max_tokens}-${dropout}-${lr}-${seed}-${warmup}-${eval_bleu}-${patience}-${GPU_NUM}

  cmd="python ${PROJECT_PATH}/fairseq_cli/train.py ${BIN_PATH}/${dataset} \
    --task $task --arch $arch \
    --source-lang $src --target-lang $tgt \
    --fp16 --num-workers 4 --seed ${seed} --dropout ${dropout} \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-updates $warmup \
    --lr $lr \
    --criterion $criterion  --label-smoothing ${label_smoothing} \
    --max-tokens $max_tokens \
    --update-freq ${update_freq} \
    --validate-interval ${validate_interval} \
    --save-dir $SAVE_DIR/$cp_name \
    --tensorboard-logdir $SAVE_DIR/$cp_name \
    --keep-last-epochs ${keep_last_epochs} \
    --patience ${patience} \
    --share-all-embeddings \
    --sent"

  if [ ${eval_bleu} -eq 1 ]; then 
  cmd=${cmd}" --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-bleu --eval-bleu-remove-bpe"
  fi 

  cmd="nohup "${cmd}" > ${LOG_DIR}/${cp_name}.log 2>&1 &"

  eval ${cmd}

  tail -f ${LOG_DIR}/${cp_name}.log

elif [ ${mode} == "doc" ]; then 

    echo `date`: train doc model

    sent_cp=$3
    sent_model=${SAVE_DIR}/${sent_cp}/checkpoint_best.pt

    max_tokens=4096

    dropout=0.1

    lr=5e-4

    patience=5

    mask_prob=0.15
    echo `date`: mask prob ${mask_prob}

    cohesion_type=(none lexical_cohesion coreference all)
    if [[ ${cohesion_type[$4]} != "none" ]]; then 
      cohesion_mask_prob=0.5
      cohesion_path=/path/to/cohesion/link
      echo `date`: cohesion mask prob ${cohesion_mask_prob}
      echo `date`: use cohesion link from ${cohesion_path}
    else
      # never apply cohesion attention mask
      cohesion_mask_prob=1.0
    fi 

    keep_last_epochs=5

    eval_bleu=1

    use_predicted_memory=1
    if [ $use_predicted_memory -eq 1 ]; then 
      echo `date`: use predicted memory, criterion label_smoothed_cross_entropy_predicted
      mu=10
      criterion=label_smoothed_cross_entropy_predicted
    else
      mu=0
    fi

    cp_name=${dataset}-${task}-${arch}-${max_tokens}-${dropout}-${lr}-${seed}-${patience}-${eval_bleu}-${mask_prob}-${cohesion_type[$4]}-${cohesion_mask_prob}-${use_predicted_memory}-${mu}-${GPU_NUM}

    predicted_memory_path=${SAVE_DIR}/${cp_name}/memory.npy 

    cmd="python ${PROJECT_PATH}/fairseq_cli/train.py ${BIN_PATH}/${dataset} \
    --task $task --arch $arch \
    --source-lang $src --target-lang $tgt \
    --fp16 --num-workers 4 --seed ${seed} --dropout ${dropout} \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-updates ${warmup} \
    --lr ${lr} \
    --criterion $criterion  --label-smoothing ${label_smoothing} \
    --max-tokens $max_tokens \
    --update-freq ${update_freq} \
    --validate-interval ${validate_interval} \
    --save-dir $SAVE_DIR/$cp_name \
    --tensorboard-logdir $SAVE_DIR/$cp_name \
    --keep-last-epochs ${keep_last_epochs} \
    --patience ${patience} \
    --finetune-from-model ${sent_model} \
    --mask-prob ${mask_prob} \
    --share-all-embeddings \
    --cohesion-path ${cohesion_path} --cohesion-mask-prob ${cohesion_mask_prob} \
    " 

    if [ ${eval_bleu} -eq 1 ]; then 
      echo `date`: use bleu as the best checkpoint metric
      cmd=${cmd}" --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-bleu --eval-bleu-remove-bpe"
    fi 


    if [ ${use_predicted_memory} -eq 1 ]; then 
    cmd=${cmd}" --predicted-memory-path ${predicted_memory_path} --mu ${mu}"
    fi  

    cmd="nohup "${cmd}" > ${LOG_DIR}/${cp_name}.log 2>&1 &"

    eval ${cmd}
    tail -f ${LOG_DIR}/${cp_name}.log

else
  echo unknown mode ${mode}, Please select correct mode.
  exit
fi