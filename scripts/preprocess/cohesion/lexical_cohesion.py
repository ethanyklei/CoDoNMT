from collections import defaultdict
import logging
import os 
import sys
import json
import argparse
from typing import List
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout
)
logger = logging.getLogger("lexical cohesion")

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="Bi-Europarl-share-fix-3-doc")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--split", type=str, default="train", choices=['train', 'valid', 'test'])
    parser.add_argument("--lang", type=str, default='en')
    parser.add_argument("--ctx-num", type=int, default=3)

    return parser.parse_args()

def read_file(path, read_lines=True):
    with open(path, 'r', encoding='utf-8') as fp:
        if read_lines:
            return fp.readlines()
        else:
            return fp.read()

def _collect_lexical_cohesion_word(words):
    result = []
    for word in words:
        synsets = wn.synsets(word)
        candicates_set = []
        for synset in synsets:
            candicates_set.extend(synset.lemma_names())
            #
            hyponyms = synset.hyponyms()
            for hyponyms_set in hyponyms:
                candicates_set.extend(hyponyms_set.lemma_names())
            # 
            hypernyms = synset.hypernyms()
            for hypernyms_set in hypernyms:
                candicates_set.extend(hypernyms_set.lemma_names())
            candicates_set = list(set(candicates_set))
        result.append(candicates_set)
    return result

def _align_words(sent, sent_offset: int = 0, bpe_delimiter="@@"):
    align_result = []
    whole_words = []
    words = sent.split()
    prev_sub_word = ""
    for word_idx, word in enumerate(words):
        if word.endswith(bpe_delimiter):
            if prev_sub_word == "":
                align_result.append([word_idx + sent_offset])
            else:
                align_result[-1].append(word_idx + sent_offset)
            prev_sub_word += word.replace(bpe_delimiter, "")
            
        else:
            if prev_sub_word != "":
                prev_sub_word += word
                whole_words.append(prev_sub_word)
                align_result[-1].append(word_idx + sent_offset)
                prev_sub_word = ""
            else:
                whole_words.append(word)
                align_result.append([word_idx + sent_offset])
    
    return align_result, whole_words


def _expect_word_filter(words, expect_type_list: List = ["PROPN", "ADJ", "NOUN"]):
    stop_words = set(stopwords.words("english"))
    expect_words = []
    expect_ids = []
    for id, word in enumerate(words):
        if word not in stop_words:
            expect_words.append(word)
            expect_ids.append(id)
    return expect_words, expect_ids


def _parse_lexical_cohesion(corpus, args):
    cohesion_device_index = defaultdict(list)
    cohesion_device_word = defaultdict(list)
    for sample_idx, sample in (enumerate(corpus)):
        if sample_idx % 10000 == 0:
            logger.info(f"parsed {sample_idx}")
        sents = sample.split(" </s> ")
        ctx_sents = sents[:-1]
        
        cur_sent_offset = sum([len(sent.split()) for sent in ctx_sents]) + args.ctx_num
        cur_sent = sents[-1]
        
        ctx_sent = " </s> ".join(ctx_sents)
        # ctx_sent = ctx_sent.replace("@@ ", "")

        cur_align_index, cur_sent_words = _align_words(cur_sent, cur_sent_offset)
        ctx_align_index, ctx_sent_words = _align_words(ctx_sent, 0)

        expect_words, expect_ids = _expect_word_filter(cur_sent_words)
        lexical_cohesion_candidates = _collect_lexical_cohesion_word(expect_words)

        for id, candidate in enumerate(lexical_cohesion_candidates):
            # for ctx_id, ctx_word in enumerate(ctx_sent_words):
            for syn_words in candidate:
                # syn word in ctx sent
                if ctx_sent.find(f" {syn_words.replace('_', ' ')} ") != -1:
                    syn_words = syn_words.split("_")
                    for ctx_id, ctx_word in enumerate(ctx_sent_words):
                        if ctx_word == syn_words[0]:
                            ctx_word = " ".join(ctx_sent_words[ctx_id: ctx_id + len(syn_words)])
                            syn_word = " ".join(syn_words)
                            if ctx_word == syn_word:
                                cohesion_device_word[sample_idx].append(
                                    [expect_words[id], ctx_word] 
                                )
                                cohesion_device_index[sample_idx].append([cur_align_index[expect_ids[id]], ctx_align_index[ctx_id]])

    return cohesion_device_index, cohesion_device_word

if __name__ == "__main__":

    args = get_args()

    corpus = read_file(os.path.join(args.data_dir, args.dataset, f"{args.split}.{args.lang}"))
    corpus = [sample.strip() for sample in corpus]

    cohesion_device_index, cohesion_device_word = _parse_lexical_cohesion(corpus=corpus, args=args)

    cohesion_device_index_path = os.path.join(args.save_dir, 'index.json')
    with open(cohesion_device_index_path, 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(cohesion_device_index))

    cohesion_device_word_path = os.path.join(args.save_dir, 'word.json')
    with open(cohesion_device_word_path, 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(cohesion_device_word))



