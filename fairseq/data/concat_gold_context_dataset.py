# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


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
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

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
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            if samples[0].get("tgt_prev", None) is not None:
                prev_output_tokens = merge(
                    "tgt_prev",
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                    pad_to_length=pad_to_length["target"]
                    if pad_to_length is not None
                    else None,
                )
            else:
                prev_output_tokens = merge(
                    "target",
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                    pad_to_length=pad_to_length["target"]
                    if pad_to_length is not None
                    else None,
                )
    else:
        ntokens = src_lengths.sum().item()
    
    prev_tgt_ctx_tokens = merge(
        "tgt_ctx",
        left_pad=False,
        move_eos_to_beginning=True,
    )

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }

    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if prev_tgt_ctx_tokens is not None:
        batch['net_input']["prev_tgt_ctx_tokens"] = prev_tgt_ctx_tokens.index_select(
            0, sort_order
        )

    return batch


class ConcatGoldContextDataset(FairseqDataset):

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
        mask_type="src",
        mask_prob=0.0,
        ctx_num=3,
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
        self.ctx_num=ctx_num
        ###########################################
    def get_batch_shapes(self):
        return self.buckets

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        tgt_prev = None
        src_item = self.src[index]

        tgt_ctx_mask = tgt_item.eq(self.tgt_dict.eos())
        tgt_ctx_idx = torch.arange(tgt_item.size(0)).masked_select(tgt_ctx_mask)
        tgt_ctx_item = tgt_item[ : tgt_ctx_idx[-2] + 1]
        tgt_item = tgt_item[tgt_ctx_idx[-2] + 1 : ]

        ##############################################################
        if self.mask_prob > 0:
            if self.mask_type == "src":
                src_item = self.soft_mask(src_item, self.src_dict, index)
            elif self.mask_type == "tgt" and tgt_item is not None:
                tgt_prev = self.soft_mask(tgt_item, self.tgt_dict, index)
            elif self.mask_type == "both":
                src_item = self.soft_mask(src_item, self.src_dict, index)
                if tgt_item is not None:
                    tgt_prev = self.soft_mask(tgt_item, self.tgt_dict, index)
        ##############################################################
        
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            ##############################
            "tgt_prev": tgt_prev, 
            "tgt_ctx": tgt_ctx_item,
            ##############################
        }

        return example

    def __len__(self):
        return len(self.src)

    ###################################################################
    def fix_mask(self, item, dictionary, index):
        mask_idx = dictionary.index("<mask>")

        assert mask_idx != dictionary.unk()

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            full_sz = len(item)

            assert (
                mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                mask_idx,
            )
            
            # decide elements to mask
            mask = np.full(full_sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * full_sz + np.random.rand()
            )
            mask[np.random.choice(full_sz, num_mask, replace=False)] = True
            
            # we do not mask the eos token
            eos_mask = item == self.eos
            mask[eos_mask] = False

            new_item = np.copy(item)
            new_item[mask] = mask_idx
            
            return torch.from_numpy(new_item)
    
    def add_noise_mask(self, item, dictionary):
        mask_idx = dictionary.index("<mask>")
        assert mask_idx != dictionary.unk()

        with data_utils.numpy_seed(self.seed):
            full_sz = len(item)

            assert (
                mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                mask_idx,
            )

            mask = np.random.rand(full_sz) < self.mask_prob
            
            new_item = np.copy(item)
            
            mask &= new_item > dictionary.nspecial
            new_item[mask] = mask_idx

        return torch.from_numpy(new_item)

    def soft_mask(self, item, dictionary, index):
        mask_idx = dictionary.index("<mask>")
        assert mask_idx != dictionary.unk()

        assert (
                mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                mask_idx,
            )

        with data_utils.numpy_seed(self.seed, self.epoch, index):

            sents_offset = np.arange(item.size()[0])[item == dictionary.eos()] 
            sents_num = len(sents_offset)

            sents_len = np.array([sents_offset[0]])

            if sents_num > 1:
                cand_sents_len = sents_offset[1:] - sents_offset[0:-1]
                cand_sents_len = cand_sents_len - 1
                sents_len = np.append(sents_len, cand_sents_len)

            # excluding </s>
            # sents_len = sents_len - 1

            full_sz = len(item)

            mask = np.full(full_sz, False)

            num_mask = np.array([self.mask_prob * sent_len + np.random.rand() for sent_len in sents_len ], dtype=np.int32)
            
            for i in range(sents_num):
                cand_mask = np.random.choice(sents_len[i], num_mask[i], replace=False)
                if i > 0:
                    cand_mask = cand_mask + sents_offset[i - 1] + 1
                mask[cand_mask] = True

            new_item = np.copy(item)
            new_item[mask] = mask_idx

            return torch.from_numpy(new_item)

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
