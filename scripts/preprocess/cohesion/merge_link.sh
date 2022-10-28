#/usr/bin/env bash 


set -u 
set -eo pipefail

TYPE=$1

DATA_DIR=${PROJECT_PATH}/log/cohesion/${DATASET}
SAVE_DIR=${PROJECT_PATH}/log/cohesion/${DATASET}/all/${TYPE}

mkdir -p ${SAVE_DIR}

python ${PROJECT_PATH}/scripts/preprocess/cohesion/merge_link.py \
                --link-dir ${DATA_DIR} \
                --save-dir ${SAVE_DIR} \


