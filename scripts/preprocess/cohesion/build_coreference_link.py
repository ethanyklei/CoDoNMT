from collections import defaultdict
import os 
import json
from unittest import result
from torch import concat
from tqdm import tqdm

import copy


import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-coref-dir", type=str, default="/home/yklei/projects/fairseq-dev/log/cohesion/Bi-Europarl-share-fix-3-doc/coreference/train/")
    parser.add_argument("--raw-data-dir", type=str, default="/home/yklei/nfs/data-prep/Bi-Europarl-share-fix-3-doc")
    parser.add_argument("--save-dir", type=str, default="/home/yklei/projects/fairseq-dev/log/cohesion/Bi-Europarl-share-fix-3-doc/coreference/train")

    return parser.parse_args()

def read_file(path, readlines: bool = True, load_json: bool = False):
    with open(path, 'r') as fp:
        if readlines:
            return fp.readlines()
        else:
            if load_json:
                return json.load(fp)
            else:
                return fp.read()

def align_word(raw_sent_words, sent_result, sample_id):
    
    align_word_index = []
    cur_bpe_word_index = 0 
    try:
        for sent_id, sent in enumerate(sent_result):
            align_word_index.append([])
            if not isinstance(cur_bpe_word_index, list):
                cur_bpe_word = raw_sent_words[cur_bpe_word_index]
            # sent_word_index = 0
            # sent_word = sent[sent_word_index]
            for word in sent:
                if cur_bpe_word == "</s>":
                    # align_word_index[-2].append([cur_bpe_word_index])
                    cur_bpe_word_index += 1
                    cur_bpe_word = raw_sent_words[cur_bpe_word_index]

                if cur_bpe_word.endswith("@@"):
                    prev_bpe_word = cur_bpe_word.replace("@@", "")
                    prev_bpe_index = [cur_bpe_word_index]
                    cur_bpe_word_index += 1
                    while raw_sent_words[cur_bpe_word_index].endswith("@@"):
                        prev_bpe_word += raw_sent_words[cur_bpe_word_index].replace("@@", "")
                        prev_bpe_index.append(cur_bpe_word_index)
                        cur_bpe_word_index += 1
                    prev_bpe_word += raw_sent_words[cur_bpe_word_index]
                    prev_bpe_index.append(cur_bpe_word_index)
                    cur_bpe_word = prev_bpe_word
                    cur_bpe_word_index = copy.copy(prev_bpe_index)

                if cur_bpe_word == word or cur_bpe_word == "\xad":
                    if isinstance(cur_bpe_word_index, list):
                        align_word_index[-1].append(cur_bpe_word_index)
                        cur_bpe_word_index = cur_bpe_word_index[-1] + 1
                    else:
                        align_word_index[-1].append([cur_bpe_word_index])
                        cur_bpe_word_index += 1
                    if cur_bpe_word_index < len(raw_sent_words):
                        cur_bpe_word = raw_sent_words[cur_bpe_word_index]
                    else:
                        assert sum([len(i) for i in align_word_index]) == sum([len(i) for i in sent_result])

                elif cur_bpe_word.find(word) == 0:
                    if isinstance(cur_bpe_word_index, list):
                        align_word_index[-1].append(cur_bpe_word_index)
                    else:
                        align_word_index[-1].append([cur_bpe_word_index])
                    cur_bpe_word = cur_bpe_word.replace(word, "", 1)
                    if cur_bpe_word == "":
                        if isinstance(cur_bpe_word_index, list):
                            cur_bpe_word_index = cur_bpe_word_index[-1] + 1
                        else:
                            cur_bpe_word_index += 1
                        cur_bpe_word = raw_sent_words[cur_bpe_word_index]                   
                else:
                    return None
    except Exception as e:
        return None
    return align_word_index

def parse_coreference(args):
    coref_align_result = defaultdict(dict)

    raw_coref_json = read_file(os.path.join(args.raw_coref_dir, "coref.json"), readlines=False, load_json=True)
    raw_sent_json = read_file(os.path.join(args.raw_coref_dir, "sent.json"), readlines=False, load_json=True)

    raw_data = read_file(os.path.join(args.raw_data_dir, 'train.en'), readlines=True)

    assert len(raw_coref_json) == len(raw_sent_json)

    total_sample_num = len(raw_data)

    unalign_sample = 0
    for sample_id in range(total_sample_num):

        sent_result = raw_sent_json[str(sample_id)]

        raw_sample = raw_data[sample_id].strip().split()

        cur_sent_offset = len(raw_sample) - raw_sample[::-1].index("</s>")

        align_word_index = align_word(raw_sample, sent_result, sample_id)

        if align_word_index == None:
            unalign_sample += 1
        else:
            for coref_links in raw_coref_json[str(sample_id)]:
                cur_word_idx = []
                align_index = []
                for sent_idx, start_idx, end_idx, word in coref_links:
                    align_index.append([])
                    sent_idx -= 1
                    start_idx -= 1
                    end_idx -= 1

                    for i in align_word_index[sent_idx][start_idx: end_idx]:
                        for j in i:
                            align_index[-1].append(j)
                
                    # assert "".join([raw_sample[i].replace("@@", "") for i in align_index[-1]]) == word, f"{sample_id} | {"".join([raw_sample[i].replace("@@", "") for i in align_index[-1]])} | {word}"       
                    align_index[-1] = list(set(align_index[-1]))
                    sorted(align_index[-1])
                    if align_index[-1][0] >= cur_sent_offset:
                        cur_word_idx.extend(align_index[-1])
                if len(cur_word_idx) > 0:
                    merge_align_index = []
                    for idx in align_index:
                        merge_align_index.extend(idx)
                    for idx in cur_word_idx:
                        coref_align_result[sample_id][idx] = copy.copy(merge_align_index)

    print(unalign_sample)

    save_path = os.path.join(args.save_dir, "link.json")

    with open(save_path, 'w') as fp:
        json.dump(coref_align_result, fp)

if __name__ == "__main__": 
    args = get_args()

    parse_coreference(args)



