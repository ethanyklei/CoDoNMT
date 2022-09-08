# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, masked_tokens_lprobs=None, src_masked_tokens=None, mask_alpha=0.2):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    #######################################
    if masked_tokens_lprobs is not None:
        if src_masked_tokens.dim() == masked_tokens_lprobs.dim() - 1:
            src_masked_tokens = src_masked_tokens.unsqueeze(-1)
        masked_tokens_nll_loss = -masked_tokens_lprobs.gather(dim=-1, index=src_masked_tokens)
        masked_tokens_smooth_nll_loss = -masked_tokens_lprobs.sum(dim=-1, keepdim=True)
    else:
        masked_tokens_nll_loss = None
        masked_tokens_smooth_nll_loss = None
    #######################################

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        ################################
        if masked_tokens_nll_loss is not None:
            masked_tokens_pad_mask = src_masked_tokens.eq(ignore_index)
            masked_tokens_nll_loss.masked_fill_(masked_tokens_pad_mask, 0.0)
            masked_tokens_smooth_nll_loss.masked_fill_(masked_tokens_pad_mask, 0.0)
        ################################
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        ######################################
        if masked_tokens_nll_loss is not None:
            masked_tokens_nll_loss = masked_tokens_nll_loss.squeeze(-1)
            masked_tokens_smooth_nll_loss = masked_tokens_smooth_nll_loss.squeeze(-1)
        ####################################
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        #######################################
        if masked_tokens_nll_loss is not None:
            masked_tokens_nll_loss = masked_tokens_nll_loss.sum()
            masked_tokens_smooth_nll_loss = masked_tokens_smooth_nll_loss.sum()
        #######################################

    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    ##############################################
    if masked_tokens_nll_loss is not None:
        masked_tokens_eps_i = epsilon / masked_tokens_lprobs.size(-1)
        masked_tokens_loss = (1.0 - epsilon) * masked_tokens_nll_loss + masked_tokens_eps_i * masked_tokens_smooth_nll_loss

        loss = loss + mask_alpha * masked_tokens_loss
    ##############################################

    return loss, nll_loss, masked_tokens_nll_loss


@register_criterion("label_smoothed_cross_entropy_mask")
class LabelSmoothedCrossEntropyMaskCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.mask_alpha = task.args.mask_alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

        parser.add_argument("--mask-alpha", type=float)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, masked_tokens_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "mask_nll_loss": masked_tokens_nll_loss.data if masked_tokens_nll_loss is not None else 0.0,
            "src_masked_ntokens": sample['src_masked_ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0: 
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_lprobs_and_mask_tokens(self, model, net_output, sample):
        masked_tokens_lprobs = utils.log_softmax(net_output[1]['encoder_predict'], dim=-1)
        src_masked_tokens = sample['src_masked_tokens']

        return masked_tokens_lprobs.view(-1, masked_tokens_lprobs.size(-1)), src_masked_tokens.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if net_output[1]['encoder_predict'] is not None:
            masked_tokens_lprobs, src_masked_tokens = self.get_lprobs_and_mask_tokens(model, net_output, sample)
        else:
            masked_tokens_lprobs = None
            src_masked_tokens = None

        loss, nll_loss, masked_tokens_nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            masked_tokens_lprobs=masked_tokens_lprobs,
            src_masked_tokens=src_masked_tokens,
            mask_alpha=self.mask_alpha
        )
        return loss, nll_loss, masked_tokens_nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ##################################################################
        mask_nll_loss_sum = sum(log.get("mask_nll_loss", 0) for log in logging_outputs)
        src_masked_ntokens = sum(log.get("src_masked_ntokens", 0) for log in logging_outputs)
        ##################################################################
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        ######################################################################
        metrics.log_scalar(
            "mask_nll_loss", mask_nll_loss_sum / src_masked_ntokens / math.log(2), src_masked_ntokens, round=3
        )
        ######################################################################
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
