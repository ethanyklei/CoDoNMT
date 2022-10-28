#/usr/bin/env bash 


set -u 
set -eo pipefail

DATASET=$1
# train valid test
TYPE=$2

DATA_DIR=/path/to/bpe/data
SAVE_DIR=/path/to/save/coreference/index/info
LOG_DIR=./log/cohesion/coreference

mkdir -p ${LOG_DIR} ${SAVE_DIR}

nohup python ./scripts/preprocess/cohesion/coreference.py \
                --dataset ${DATASET} \
                --data-dir ${DATA_DIR} \
                --save-dir ${SAVE_DIR} \
                --split ${TYPE} \
                --num-workers 20 > ${LOG_DIR}/parse_coreference.log 2>&1 &