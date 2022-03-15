# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Tuple, Callable

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process
from fairseq.models.speech_to_text.utils import (
    get_prob,
    save_to_dict
)

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss

def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats

def sigmoid(x):
  return 1. / (1. + math.exp(-x))

@register_criterion("label_smoothed_cross_entropy_with_stmm_self_learning")
class LabelSmoothedCrossEntropyCriterionWithSTMMSelfLearning(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, jsd_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing)
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.report_accuracy = True
        self.jsd_weight = jsd_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--jsd-weight",
            default=1.0,
            type=float,
            metavar="D",
            help="weight of jsd loss"
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert model.encoder.task == "stack" and model.encoder.mixup
        audio, audio_lengths, source, source_lengths, prev_output_tokens, align_pad, align_lengths = sample["net_input"].values()
        audio, audio_encoder_padding_mask = model.encoder.encode_audio(audio, audio_lengths)
        source, source_encoder_padding_mask = model.encoder.forward_embedding(source)
        max_prob = get_prob(model.encoder.num_updates, model.encoder.mixup_arguments, model.training)
        # st loss
        source_st, source_encoder_padding_mask_st = model.encoder.get_mixed_input(audio, source, align_pad, align_lengths, 1.0)
        encoder_out_st, encoder_padding_mask_st = model.encoder.encode_text(source_st, source_encoder_padding_mask_st)
        encoder_out_st = save_to_dict(encoder_out_st, encoder_padding_mask_st)
        net_output_st = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out_st
        )
        loss_st, nll_loss_st, lprobs_st, target_st = self.compute_loss_with_lprobs(model, net_output_st, sample, reduce=reduce)
        probs = [0.0]
        if model.training:
            # mixup loss
            probs = self.get_mixup_probs(model, net_output_st, sample, self.padding_idx, max_prob)
            source_mix, source_encoder_padding_mask_mix = model.encoder.get_mixed_input(audio, source, align_pad, align_lengths, probs)
            encoder_out_mix, encoder_padding_mask_mix = model.encoder.encode_text(source_mix, source_encoder_padding_mask_mix)
            encoder_out_mix = save_to_dict(encoder_out_mix, encoder_padding_mask_mix)
            net_output_mix = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out_mix
            )
            loss_mix, nll_loss_mix, lprobs_mix, target_mix = self.compute_loss_with_lprobs(model, net_output_mix, sample, reduce=reduce)
            # output jsd loss
            output_jsd = self.compute_jsd_loss(lprobs_st, lprobs_mix, target_st, target_mix, self.padding_idx)
        else:
            loss_mix, nll_loss_mix, output_jsd = torch.tensor([0.]).cuda(), torch.tensor([0.]).cuda(), torch.tensor([0.]).cuda()
        # log
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "trans_loss": utils.item(loss_st.data) if reduce else loss_st.data,
            "nll_loss": utils.item(nll_loss_st.data) if reduce else nll_loss_st.data,
            "trans_loss_mix": utils.item(loss_mix.data) if reduce else loss_mix.data,
            "nll_loss_mix": utils.item(nll_loss_mix.data) if reduce else nll_loss_mix.data,
            "output_jsd": utils.item(output_jsd.data) if reduce else output_jsd.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "prob": sum(probs) / len(probs),
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output_st, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        loss = loss_st + loss_mix + output_jsd * self.jsd_weight
        logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        return loss, sample_size, logging_output

    def compute_jsd_loss(self, lprobs_st, lprobs_mix, target_st, target_mix, ignore_index):
        kl_loss_st = F.kl_div(lprobs_mix, lprobs_st, log_target=True, reduction="none").sum(-1)
        kl_loss_mix = F.kl_div(lprobs_st, lprobs_mix, log_target=True, reduction="none").sum(-1)
        pad_mask = target_st.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        pad_mask = target_mix.eq(ignore_index)
        kl_loss_mix.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mix = kl_loss_mix.sum()
        kl_loss = (kl_loss_st + kl_loss_mix) / 2.0
        return kl_loss

    def compute_loss_with_lprobs(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target

    def compute_acc(self, lprobs, target, ignore_index):
        _, pred = torch.max(lprobs, dim=-1)
        idx = ~target.eq(ignore_index)
        pred = pred[idx]
        target = target[idx]
        acc = (torch.sum(pred == target) / len(target)).detach().item()
        return acc

    def compute_uncertainty(self, lprobs, target, ignore_index):
        upper = math.log(lprobs.shape[-1])
        entropy = -torch.sum((torch.exp(lprobs) * lprobs), dim=-1)
        idx = ~target.eq(ignore_index)
        entropy = torch.mean(entropy[idx]) / upper
        return entropy.detach().item()

    def get_mixup_probs(self, model, net_output_st, sample, ignore_index, max_prob):
        lprobs = model.get_normalized_probs(net_output_st, log_probs=True)
        target = model.get_targets(sample, net_output_st)
        bsz = len(target)
        probs = [self.compute_uncertainty(lprobs[i], target[i], ignore_index) * max_prob for i in range(bsz)]
        probs = [sigmoid(prob - 0.5) for prob in probs]
        return probs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        trans_loss_mix_sum = utils.item(
            sum(log.get("trans_loss_mix", 0) for log in logging_outputs)
        )
        nll_loss_mix_sum = utils.item(
            sum(log.get("nll_loss_mix", 0) for log in logging_outputs)
        )
        output_jsd_sum = utils.item(
            sum(log.get("output_jsd", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        prob = utils.item(max(log.get("prob", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "trans_loss", trans_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "trans_loss_mix", trans_loss_mix_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss_mix", nll_loss_mix_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "output_jsd", output_jsd_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "prob", prob, 1, round=3
        )
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