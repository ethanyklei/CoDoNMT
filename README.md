# CoDoNMT: Modeling Cohesion Devices for Document-Level Neural Machine Translation
This is an implementation of COLING 2022 paper "CoDoNMT: Modeling Cohesion Devices for Document-Level Neural Machine Translation"

Our implement is based on the [Fairseq](https://github.com/pytorch/fairseq).


# Requirements and Installation
- PyTorch >= 1.7
- Python >= 3.8
- sacrebleu=1.5.1
you
```
git clone git@github.com:codeboy311/CoDoNMT.git
cd CoDoNMT
pip install -e .
```

# Data Preprocess
We use Open16 as example to show how to preprocess data.

## Fairseq preprocess

We prepare both sent-level data and doc-level data. The placeholder of `preprocess-Bi-LSTM-Open-en2ru.sh` is:
- ctx_num [int]: the number of context sentence.
- doc_only [int]: whether prepare sent-level data, default is 0.

```
bash ./scripts/preprocess/data/preprocess-Bi-LSTM-Open-en2ru.sh 3 0
```

## Cohesion preparation
We first detect lexical cohesion devices and coreference, separately.

For lexical cohesion devices detection:
```
bash ./scripts/preprocess/cohesion/lexical_cohesion.sh
```
For coreference detection:
```
bash ./scripts/preprocess/cohesion/coreference.sh
```

Then we construct coheison links according to the location information of cohesive devices.

For lexical cohesion links:
```
bash ./scripts/preprocess/cohesion/build_cohesion_link.sh
```

For coreference links:
```
bash ./scripts/preprocess/cohesion/build_coreference_link.sh
```

Finally, we merge two types of cohesion links:
```
bash ./scripts/preprocess/cohesion/merge_link.sh
```

# Training
Our model are trained with two stages.

First, Using sent-level paralle data to train the sent-level pre-train model.
```
bash ./scripts/train/train-concat.sh sent 0,1,2,3 
```
Then, The doc-level model is initialized with the sent-level pre-train model and trained with doc-level data.
```
bash ./scripts/train/train-concat.sh sent 0,1,2,3 /path/to/sent-level/model/checkpoint/root
```

# Inference
To generate and evaluate your model.
```
python fairseq_generate ${BIN_PATH} \
        --path ${CP_DIR}/${cp_name}/checkpoint_best.pt \
        --source-lang en --target-lang ru \
        --max-len-a 1.2 --max-len-b 10 \
        --lenpen 0.7 \
        --task translation_doc_concat \
        --beam 4 \
        --batch-size 32 \
        --remove-bpe
```


# Citation
```
```