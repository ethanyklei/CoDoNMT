import json
from stanfordcorenlp import StanfordCoreNLP
import logging
import os 
import sys
import copy
from tqdm import tqdm

from urllib.parse import ParseResult, quote
from collections import defaultdict

from multiprocessing import Pool
import argparse


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout
)
logger = logging.getLogger("generate_mask")

# you need to set up your personal path to stanford corenlp package.
nlp = StanfordCoreNLP('/home/toolkits/stanford-corenlp-4.4.0', lang="en", logging_level=logging.INFO)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Bi-Europarl-share-fix-3-doc")
    parser.add_argument("--data-dir", type=str, default="/home/yklei/nfs/data-prep")
    parser.add_argument("--save-dir", type=str, default="/home/yklei/projects/fairseq-dev/log/coreference/Bi-Europarl-share-fix-3-doc")
    parser.add_argument("--split", type=str, choices=['train', 'valid', 'test'], default="train")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--merge", action="store_true", default=False)

    args = parser.parse_args()

    return args

def _coreference(args, worker_id, samples, prefix):
    sample_num = len(samples)
    logger.info(f"start {worker_id} | sent num: {sample_num} ")

    coref_dict = defaultdict(list)
    coref_sents = defaultdict(list)
    for sample_id in (range(sample_num)):
        if sample_id % 100 == 0:
            logger.info(f"{worker_id}: {sample_id}/{sample_num} | {sample_id / sample_num}")
            
        sample = samples[sample_id].strip()
        
        sample = sample.replace("@@ ", "")

        sample = sample.replace(" </s>", "")

        try:   
            corefs, sents = nlp.coref(quote(sample))
            coref_dict[sample_id] = corefs
            coref_sents[sample_id] = sents
        except Exception as e: 
            logger.error(e)
            nlp.close()

    tmp_dir = os.path.join(args.save_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    raw_corefs_file = os.path.join(tmp_dir, f"{prefix}-{worker_id}-raw.json")
    corefs_sents_file = os.path.join(tmp_dir, f"{prefix}-{worker_id}-sents.json")

    with open(raw_corefs_file, 'w') as fp:
        fp.write(json.dumps(coref_dict))
    with open(corefs_sents_file, 'w') as fp:
        fp.write(json.dumps(coref_sents))


def parse_tokens(sents):
    tokens = []
    for sent in sents:
        tokens.append([])
        for token in sent['tokens']:
            tokens[-1].append(token['word'])

    return tokens

def merge_result(args):
    tmp_dir = os.path.join(args.save_dir, "tmp")
    files = os.listdir(tmp_dir)
    sent_files = list(filter(lambda x: x.endswith("sents.json"), files))
    sent_files = sorted(sent_files, key=lambda x: int(x.split("-")[-2]))
    coref_files = list(filter(lambda x: x.endswith("raw.json"), files))
    coref_files = sorted(coref_files, key=lambda x: int(x.split("-")[-2]))
    coref_merge_result = defaultdict(list)
    sent_merge_result = defaultdict(dict)

    offset = 0
    for i in tqdm(range(len(coref_files))):
        with open(os.path.join(tmp_dir, coref_files[i]), 'r') as fp:
            corefs = json.load(fp)
        with open(os.path.join(tmp_dir, sent_files[i]), 'r') as fp:
            sents = json.load(fp)
        assert len(corefs) == len(sents)
        for coref_idx in corefs.keys():
            coref_merge_result[int(coref_idx) + offset] = corefs[coref_idx]
            sent_merge_result[int(coref_idx) + offset] = parse_tokens(sents[coref_idx])
        
        offset += len(corefs)

    with open(os.path.join(args.save_dir, "coref.json"), 'w') as fp:
        fp.write(json.dumps(coref_merge_result))

    with open(os.path.join(args.save_dir, "sent.json"), 'w') as fp:
        fp.write(json.dumps(sent_merge_result))


if __name__ == "__main__":

    args = get_args()

    with open(os.path.join(args.data_dir, args.dataset, f"{args.split}.en"), 'r') as fp:
        corpus = fp.readlines()

    total_sample_num = len(corpus)

    split_num = total_sample_num // args.num_workers

    pool = Pool(processes=args.num_workers)
    for worker_id in range(args.num_workers):
        start_index = worker_id * split_num 
        end_index = (worker_id + 1) * split_num
        prefix = f"{args.dataset}-{args.split}"
        pool.apply_async(
            _coreference,
            (
                args,
                worker_id,
                corpus[start_index: end_index] if worker_id != args.num_workers - 1 else corpus[start_index:],
                prefix,
            )
        )

    pool.close()
    pool.join()

    nlp.close()

    merge_result(args)

    logger.info("finish !!!")


    
