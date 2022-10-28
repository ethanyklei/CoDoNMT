#/usr/bin/env bash 

set -u 
set -eo pipefail

ctx_num=$1

doc_only=$2

src=en
tgt=ru
lang_pair=${src}-${tgt}

DATA_DIR=path/to/raw/data
PREP=/prep/data/path 
SENT_PREP=${PREP}/Bi-Open16-share-sent 
PART_DOC_PREP=${PREP}/Bi-Open16-share-fix-${ctx_num}-doc

SENT_TMP=${SENT_PREP}/tmp 
PART_DOC_TMP=${PART_DOC_PREP}/tmp

mkdir -p ${SENT_TMP} ${PART_DOC_TMP}

BIN_PATH=/path/to/bin/data
SENT_BIN=${BIN_PATH}/Bi-Open16-share-sent
PART_DOC_BIN=${BIN_PATH}/Bi-Open16-share-fix-${ctx_num}-doc

MOSES=/path/to/moses
TK=${MOSES}/tokenizer/tokenizer.perl
TRAIN_TC=${MOSES}/recaser/train-truecaser.perl
TC=${MOSES}/recaser/truecase.perl
SUBWORD=/path/to/subword
NUM_OPERATIONs=30000

if [ ${doc_only} -eq 0 ]; then

    for split in train test dev; do 
        for L in English Russian; do

            if [ $L == English ]; then 
                dst_L=$src 
            else
                dst_L=$tgt
            fi

            if [ $split == dev ]; then 
                dst_split=valid
            else 
                dst_split=$split
            fi  

            cat ${DATA_DIR}/concatenated_${L}_${split}.txt | \
            sed -e "s/<sp>//g" | \
            sed -e "/^\s*$/d" | \
            sed -e "s/<en> //g" | \
            sed -e "s/<2en> //g" | \
            sed -e "s/<d>//g" | \
            perl ${TK} -threads 8 -l ${dst_L} > ${SENT_TMP}/${dst_split}.${dst_L}.tok

        done

    done

    src_truecase_model=${SENT_TMP}/truecase.${src}.mdl
    echo `date`: train truecase model to $src_truecase_model
    perl ${TRAIN_TC} --model ${src_truecase_model} --corpus ${SENT_TMP}/train.${src}.tok 

    tgt_truecase_model=${SENT_TMP}/truecase.${tgt}.mdl
    echo `date`: train truecase model to $tgt_truecase_model
    perl ${TRAIN_TC} --model ${tgt_truecase_model} --corpus ${SENT_TMP}/train.${tgt}.tok 

    echo `date`: apply truecase
    for split in train valid test; do
        perl ${TC} --model $src_truecase_model < ${SENT_TMP}/${split}.${src}.tok > ${SENT_TMP}/${split}.${src}.tok.tc
        perl ${TC} --model $tgt_truecase_model < ${SENT_TMP}/${split}.${tgt}.tok > ${SENT_TMP}/${split}.${tgt}.tok.tc
    done

    bpe_code=${SENT_TMP}/${lang_pair}.bpe.code

    learn_bpe_corpus=${SENT_TMP}/train.$lang_pair.bpe
    cat ${SENT_TMP}/train.$src.tok.tc > $learn_bpe_corpus
    cat ${SENT_TMP}/train.$tgt.tok.tc >> $learn_bpe_corpus

    echo `date`: learn bpe on $learn_bpe_corpus
    python ${SUBWORD}/learn_bpe.py -s $NUM_OPERATIONs < $learn_bpe_corpus > $bpe_code

    for L in $src $tgt; do 
        for split in train valid test; do
            echo `date`: apply_bpe.py to ${SENT_TMP}/${split}.$L.tok.tc...
            python ${SUBWORD}/apply_bpe.py -c $bpe_code < ${SENT_TMP}/${split}.$L.tok.tc > ${SENT_TMP}/${split}.$L.tok.tc.bpe
        done
    done

    echo `date`: apply doc-level special tags ...
    for L in $src $tgt; do
        for F in train.$L valid.$L test.$L; do
            cat ${SENT_TMP}/$F.tok.tc.bpe | \
            # replace empty line with [DOC]
            sed -e 's/^$/[DOC]/g' | \
            # connect all lines into one line
            sed -z -e 's/\n/ [SEP] /g' | \
            # replace the begin of doc with newline
            sed -e 's/ \[DOC\] \[SEP\] /\n/g' | \
            # handle the begin-symbol of the first doc
            sed -e 's/\[DOC\] \[SEP\] //g' | \
            # replace all [SEP] with </s>
            sed -e 's/\[SEP\]/<\/s>/g' > ${SENT_TMP}/$F
        done
    done

    python ./fairseq_cli/sample_builder.py \
        --src-lang ${src} --tgt-lang ${tgt} \
        --data-dir ${SENT_TMP} \
        --save-dir ${SENT_PREP} \
        --sent
fi

python ./fairseq_cli/sample_builder.py \
    --src-lang ${src} --tgt-lang ${tgt} \
    --data-dir ${SENT_TMP} \
    --save-dir ${PART_DOC_PREP} \
    --ctx-num ${ctx_num}

python ./fairseq_cli/preprocess.py \
    --source-lang ${src} --target-lang ${tgt} \
    --workers 16 \
    --trainpref ${SENT_PREP}/train --validpref ${SENT_PREP}/valid --testpref ${SENT_PREP}/test \
    --destdir ${SENT_BIN} \
    --joined-dictionary


python ./fairseq_cli/preprocess.py \
    --source-lang ${src} --target-lang ${tgt} \
    --workers 16 \
    --trainpref ${PART_DOC_PREP}/train --validpref ${PART_DOC_PREP}/valid --testpref ${PART_DOC_PREP}/test \
    --destdir ${PART_DOC_BIN} \
    --srcdict ${SENT_BIN}/dict.${src}.txt --tgtdict ${SENT_BIN}/dict.${tgt}.txt