#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
import logging
import os
from select import select
from statistics import mode
import sys
from itertools import chain
from matplotlib.pyplot import axis

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("create datastore")


def main(args, override_args=None):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)
    
    dataset_idx = 1
    dstore_idx = 0
    for subset in args.valid_subset.split(","):
        try:
            ###############################################
            task.args.required_seq_len_multiple = 1
            task.args.load_alignments = False
            task.load_dataset(subset, combine=False, epoch=dataset_idx)
            dataset_idx += 1 
            ##############################################
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        match_num = 0
        unmatch_num = 1
        #############################################
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample

                target = sample['target']

                batch_size, seq_len = target.size()

                features = task.forward_and_get_hidden_state_step(sample, model)

                lprobs = model.decoder.output_projection(features)

                lprobs = model.get_normalized_probs([lprobs], log_probs=True)

                top_prediction_score, top_prediction_index = torch.topk(
                    lprobs,
                    k=1
                ) # [bsz, seq_len]

                top_prediction_index = top_prediction_index.reshape(batch_size, -1).contiguous()

                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)
                
                match_mask = (target ==  top_prediction_index) & target_mask
                unmatch_mask = (target != top_prediction_index) & target_mask

                # # remove the pad tokens and related hidden states
                target = target.view(batch_size * seq_len) 
                # target_mask = target_mask.view(batch_size * seq_len)
                match_mask = match_mask.view(batch_size * seq_len)
                match_num += torch.sum(match_mask)
                match_index = match_mask.nonzero().squeeze(-1)

                unmatch_mask = unmatch_mask.view(batch_size * seq_len)
                unmatch_num += torch.sum(unmatch_mask)
                unmatch_index = unmatch_mask.nonzero().squeeze(-1)
                
                target = target.index_select(dim=0, index=match_index)
                
                features = features.contiguous().view(batch_size * seq_len, -1)
                features = features.index_select(dim=0, index=match_index)

                current_batch_count = target.size(0)

                if current_batch_count + dstore_idx > args.dstore_size:
                    reduce_size = args.dstore_size - dstore_idx
                    features = features[:reduce_size]
                    target = target[:reduce_size]
                else:
                    reduce_size = current_batch_count

                dstore_idx += reduce_size

            logger.info(f"{dstore_idx}-{args.dstore_size}")

            logger.info(f"match token num: {match_num}")
            logger.info(f"unmatch token num: {unmatch_num}")
            logger.info(f"unmatch ratio: {unmatch_num / args.dstore_size}")


def cli_main():
    parser = options.get_create_datastore_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_create_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
