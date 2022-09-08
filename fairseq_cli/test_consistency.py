#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from itertools import chain

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
logger = logging.getLogger("fairseq_cli.validate")


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

    # Build criterion
    # criterion = task.build_criterion(model_args)
    # criterion.eval()

    consistency_name = ["deixis_test", "lex_cohesion_test", "ellipsis_infl", "ellipsis_vp"]
    consistency_index = 0
    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
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

        
        with torch.no_grad():
            scores = []
            ids = []
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                
                net_output = model(**sample["net_input"])
                
                lprobs = model.get_normalized_probs(net_output, log_probs=True) 
                target = model.get_targets(sample, net_output)     



                bsz, seq_len = target.size()

                eos_mask = target.eq(task.tgt_dict.eos())
                ctx_offset = torch.arange(seq_len, device=eos_mask.device).repeat(bsz, 1).masked_select(eos_mask)
                ctx_offset = ctx_offset.view(bsz, -1)

                ctx_mask = torch.zeros((bsz, seq_len), device=eos_mask.device)
                for i in range(bsz):
                    ctx_mask[i][:ctx_offset[i][-2] + 1] = 1
                ctx_mask = ctx_mask.bool()

                
                # lprobs = lprobs.view(-1, lprobs.size(-1)) # B x T x V
                # target = target.view(-1) # B x T
                target = target.unsqueeze(-1)

                nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
                # smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)

                pad_mask = target.eq(task.tgt_dict.pad()).squeeze(-1)
                mask = pad_mask | ctx_mask
                nll_loss.masked_fill_(mask, 0.0)
                # smooth_loss.masked_fill_(mask, 0.0)

                nll_loss = nll_loss.sum(-1)
                # smooth_loss = smooth_loss.sum(-1)

                # eps_i = 1 / lprobs.size(-1)
                # loss = (1.0 - 0.1) * nll_loss + eps_i * smooth_loss
                loss = nll_loss
                # eps_i = 1e-8 / lprobs.size(-1)
                # loss = (1.0 - 1e-8) * nll_loss + eps_i * smooth_loss

                scores.extend(loss.cpu().tolist())
                ids.extend(sample['id'].cpu().tolist())

            sort_order = np.argsort(ids)
            ids = np.array(ids)
            scores = np.array(scores)
            ids = ids[sort_order]
            scores = scores[sort_order]
            with open(os.path.join(args.results_path, consistency_name[consistency_index]), 'w') as fp:
                for score in scores:
                    fp.write(f"{score}\n")

            # score_file.close()

                # _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)

                # progress.log(log_output, step=i)
                # log_outputs.append(log_output)

        consistency_index += 1

def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
