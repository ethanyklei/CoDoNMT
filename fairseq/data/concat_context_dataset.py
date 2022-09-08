# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import replace
from lib2to3.pgen2 import token
import logging
from random import sample
from time import time

import numpy as np
from regex import R
import torch
from fairseq.data import FairseqDataset, data_utils
import copy
import os


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def pad_cohesion_link(key, tokens):
        cohesion_links = []        
        bsz, seq_len = tokens.size()
        for i in range(bsz):
            cohesion_links.append(defaultdict(list))
            pad_offset = tokens[i].eq(pad_idx).sum().item()
            if samples[i][key] is not None:
                for word_idx in samples[i][key]:
                    cohesion_links[-1][int(word_idx) + pad_offset] = np.array(samples[i][key][word_idx], dtype=np.int32) + pad_offset

        return cohesion_links

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )

    masked_src = merge(
        "masked_source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )

    #################################################################
    cohesion_links = None
    if 'cohesion_links' in samples[0]:
        cohesion_links = pad_cohesion_link('cohesion_links', src_tokens)
    #################################################################

    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    masked_src = masked_src.index_select(0, sort_order)

    ###############################################
    sorted_cohesion_links = None
    if cohesion_links is not None:
        sorted_cohesion_links = []
        for sorted_id in sort_order:
            sorted_cohesion_links.append(cohesion_links[sorted_id])
    ###############################################

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            prev_output_tokens = merge(
                "tgt_prev",
                left_pad=left_pad_target,
                move_eos_to_beginning=True
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        "masked_src": masked_src,
    }

    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )
    
    ##########################################
    if sorted_cohesion_links is not None:
        batch['net_input']['cohesion_links'] = sorted_cohesion_links
    ##########################################

    return batch


class ConcatContextDataset(FairseqDataset):

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        ######################
        seed=None,
        mask_type=None,
        mask_prob=0.0,
        cohesion=None,
        predicted_memory_path=None,
        mu=5,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        

        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple
        ###########################################
        self.seed = seed
        self.mask_type = mask_type
        self.mask_prob = mask_prob
        self.epoch = 0
        self.cohesion = cohesion
        self.predicted_memory = None
        self.mu = 0
        self.sub_prob = 1.0
        if predicted_memory_path is not None:
            self.max_tgt_size = max(self.tgt_sizes)
            self.predicted_memory = np.memmap(predicted_memory_path, mode='w+', dtype=np.int32, shape=(len(self.tgt), self.max_tgt_size))
            self.mu = mu
        ###########################################
    def get_batch_shapes(self):
        return self.buckets

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
        ##################################################################
        if self.predicted_memory is not None:
            self.sub_prob = self.mu / (self.mu + np.exp((self.epoch - 1) / self.mu))
            logger.info(f"set sub prob to {self.sub_prob}")
        #################################################################

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        tgt_prev = tgt_item.clone()

        ##############################################################
        cohesion_links = None
        if self.cohesion is not None and str(index) in self.cohesion:
            cohesion_links = self.cohesion[str(index)]

        
        if self.predicted_memory is not None and self.epoch > 1:
            with data_utils.numpy_seed(self.seed, self.epoch, index):
                for i in range(len(tgt_prev)):
                    if tgt_prev[i] != self.tgt_dict.eos():
                        if np.random.random() > self.sub_prob:
                            tgt_prev[i] = self.predicted_memory[index][i]

        if self.mask_prob > 0:
            # if self.mask_type == "random-full":
            #     src_item, _ = self.add_soft_mask(src_item, self.src_dict, index, full=True)
            # elif self.mask_type == "random-cur":
            #     src_item, _ = self.add_soft_mask(src_item, self.src_dict, index)
            # elif self.mask_type == "cohesion-full":
            src_item, masked_src = self.add_cohesion_mask(src_item, list(cohesion_links.keys()) if cohesion_links is not None else None, self.src_dict, index, full=True)

        ##############################################################
        
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            ##############################
            "tgt_prev": tgt_prev, 
            "cohesion_links": cohesion_links,
            "masked_source": masked_src,
            ##############################
        }
        return example

    def __len__(self):
        return len(self.src)

    ###################################################################
    def add_soft_mask(self, item, dictionary, index, full: bool = False):
        mask_idx = dictionary.index("<mask>")
        assert mask_idx != dictionary.unk()

        # assert (
        #         mask_idx not in item
        #     ), "Dataset contains mask_idx (={}), this is not expected!".format(
        #         mask_idx,
        #     )

        with data_utils.numpy_seed(self.seed, self.epoch, index):

            sents_offset = np.arange(item.size()[0])[item == dictionary.eos()] + 1
            sents_num = len(sents_offset)

            sents_len = np.array([sents_offset[0]]) - 1

            if sents_num > 1:
                cand_sents_len = sents_offset[1:] - sents_offset[0:-1]
                cand_sents_len = cand_sents_len - 1
                sents_len = np.append(sents_len, cand_sents_len)

            full_sz = len(item)

            mask = np.full(full_sz, False)

            if full:
                num_mask = np.array([self.mask_prob * sent_len + np.random.rand() for sent_len in sents_len ], dtype=np.int32)
                
                for i in range(sents_num):
                    cand_mask = np.random.choice(sents_len[i], num_mask[i], replace=False)
                    if i > 0:
                        cand_mask = cand_mask + sents_offset[i - 1]
                    mask[cand_mask] = True
            else:
                num_mask = int(self.mask_prob * sents_len[-1] + np.random.rand())

                cand_mask = np.random.choice(sents_len[-1], num_mask, replace=False)
                cand_mask = cand_mask + sents_offset[-2]
                mask[cand_mask] = True


            new_item = np.copy(item)
            new_item[mask] = mask_idx

            masked_tokens = np.full(new_item.shape, dictionary.pad())
            masked_tokens[mask] = item[mask]

            return torch.from_numpy(new_item), torch.from_numpy(masked_tokens)
    
    def add_cohesion_mask(self, item, cohesion_words, dictionary, index, full: bool = False):
        mask_idx = dictionary.index("<mask>")
        assert mask_idx != dictionary.unk()

        if cohesion_words is not None:
            cohesion_words = [int(word) for word in cohesion_words]

        assert (
                mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                mask_idx,
            )

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            sents_offset = np.arange(item.size()[0])[item == dictionary.eos()] + 1
            sents_num = len(sents_offset)

            sents_len = np.array([sents_offset[0]]) - 1

            if sents_num > 1:
                cand_sents_len = sents_offset[1:] - sents_offset[0:-1]
                cand_sents_len = cand_sents_len - 1
                sents_len = np.append(sents_len, cand_sents_len)

            full_sz = len(item)

            mask = np.full(full_sz, False)

            if full:
                num_mask = np.array([self.mask_prob * sent_len + np.random.rand() for sent_len in sents_len ], dtype=np.int32)
                
                for i in range(sents_num - 1):
                    cand_mask = np.random.choice(sents_len[i], num_mask[i], replace=False)
                    if i > 0:
                        cand_mask = cand_mask + sents_offset[i - 1]
                    mask[cand_mask] = True
            
            if cohesion_words is not None:
                cand_mask = copy.copy(cohesion_words)
                if len(cand_mask) < num_mask[-1]:
                    remain = num_mask[-1] - len(cand_mask)
                    shuffle_index = np.arange(sents_len[-1]) + sents_offset[-2]
                    np.random.shuffle(shuffle_index)
                    for i in shuffle_index:
                        if i not in cand_mask:
                            cand_mask.append(i)
                            remain -= 1
                            if remain == 0:
                                break
                elif len(cand_mask) > num_mask[-1]:
                    cand_mask = np.array(cand_mask, dtype=np.int32)
                    cand_mask = cand_mask[np.random.choice(len(cand_mask), num_mask[-1], replace=False)]
                else:
                    cand_mask = copy.copy(cohesion_words)
                assert len(cand_mask) >= num_mask[-1]
                mask[cand_mask] = True
            else:
                cand_mask = np.random.choice(sents_len[-1], num_mask[-1], replace=False)
                cand_mask = cand_mask + sents_offset[-2]
                mask[cand_mask] = True
                        
            new_item = np.copy(item)
            new_item[mask] = mask_idx

            masked_tokens = np.full(new_item.shape, dictionary.pad())
            masked_tokens[mask] = item[mask]

            return torch.from_numpy(new_item), torch.from_numpy(masked_tokens)
    ###################################################################

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
