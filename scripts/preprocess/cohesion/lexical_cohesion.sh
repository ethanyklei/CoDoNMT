#/usr/bin/env bash 


set -u 
set -eo pipefail

DATASET=$1
# train valid test
TYPE=$2

DATA_DIR=/path/to/bpe/data
SAVE_DIR=/path/to/save/lexical_cohesion/index/info
LOG_DIR=./log/cohesion/lexical_cohesion

mkdir -p ${LOG_DIR} ${SAVE_DIR}

nohup python ./scripts/preprocess/cohesion/lexical_cohesion.py \
                --dataset ${DATASET} \
                --data-dir ${DATA_DIR} \
                --save-dir ${SAVE_DIR} \
                --split ${TYPE} > ${LOG_DIR}/parse_lexical_cohesion.log 2>&1 &


