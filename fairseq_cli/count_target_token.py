import os 
import argparse

import torch 

from fairseq.data import (
    Dictionary,
    data_utils,
    LanguagePairDataset,
    IndexedDataset,
    ConcatContextDataset,
)


def load_dictionary(path):
    return Dictionary.load(path)

def encode_line(dictionary, toks):
    line = [dictionary[tok] for tok in toks]
    return " ".join(line)



if __name__ == "__main__":
    # SAC-IWSLT2017-doc Voita-2019-sent Voita-2019-doc IWSLT15-share-3-doc

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--data-dir", type=str, default="/home/yklei/nfs/data-bin")
    parser.add_argument("--src-lang", type=str, default="zh")
    parser.add_argument("--tgt-lang", type=str, default="en")

    args = parser.parse_args()

    src_dict = load_dictionary(os.path.join(args.data_dir, args.dataset, f'dict.{args.src_lang}.txt'))
    tgt_dict = load_dictionary(os.path.join(args.data_dir, args.dataset, f'dict.{args.tgt_lang}.txt'))

    src_dataset = data_utils.load_indexed_dataset(
           os.path.join(args.data_dir, args.dataset, f"train.{args.src_lang}-{args.tgt_lang}.{args.src_lang}") , src_dict, "mmap"
    )
    tgt_dataset = data_utils.load_indexed_dataset(
           os.path.join(args.data_dir, args.dataset, f"train.{args.src_lang}-{args.tgt_lang}.{args.tgt_lang}") , tgt_dict, "mmap"
    )

    tgt_cur_tokens_num = 0
    for tgt_item in tgt_dataset:

        eos_mask = tgt_item.eq(tgt_dict.eos())
        eos_idx = torch.arange(len(tgt_item)).masked_select(eos_mask)
        tgt_cur_tokens_num += eos_idx[-1] - eos_idx[-2]
    print(tgt_cur_tokens_num.item())
        
