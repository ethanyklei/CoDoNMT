#/usr/bin/env bash 

set -u 
set -eo pipefail


INDEX_PATH=/path/to/lexical_cohesion/info
SAVE_DIR=/path/to/save/lexical_cohesion/link/info

python ./scripts/preprocess/cohesion/build_cohesion_link.py \
        --index-path ${INDEX_PATH} \
        --save-dir ${SAVE_DIR}