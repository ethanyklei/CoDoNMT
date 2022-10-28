#/usr/bin/env bash 


set -u 
set -eo pipefail


RAW_DATA_DIR=/path/to/bpe/data
RAW_COREF_DIR=/path/to/coreference/index/info
SAVE_DIR=/path/to/save/coreference/link/info

mkdir -p ${SAVE_DIR}

python ./scripts/preprocess/cohesion/build_coreference_link.py \
                --raw-coref-dir ${RAW_COREF_DIR} \
                --raw-data-dir ${RAW_DATA_DIR} \
                --save-dir ${SAVE_DIR} 
