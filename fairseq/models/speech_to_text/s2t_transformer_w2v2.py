#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, OrderedDict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    MultiheadAttention,
)
from fairseq.models.speech_to_text.utils import (
    get_prob,
    mix_input,
    save_to_dict,
)
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoderScriptable, S2TTransformerEncoder
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from examples.speech_to_text.hdfs_utils import torchHLoad
from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("s2t_transformer_w2v2")
class S2TTransformerModelW2V2(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--translation-encoder-layers", type=int, metavar="N", help="num translation encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-mt-encoder-decoder-from",
            type=str,
            metavar="STR",
            help="model to take mt encoder/decoder weights from (for initialization)",
        )
        parser.add_argument(
            '--encoder-freezing-updates',
            type=int,
            metavar='N',
            help='freeze encoder for first N updates'
        )
        parser.add_argument(
            "--mixup",
            action="store_true",
            help="if mix input of translation encoder"
        )
        parser.add_argument(
            "--mixup-arguments",
            type=str,
            metavar="STR",
            help="arguments for adjusting the probability p of mixup"
        )
        # wav2vec2 arguments
        parser.add_argument(
            "--w2v2-model-path",
            type=str,
            metavar="STR",
            help="path/to/wav2vec/model"
        )
        parser.add_argument(
            "--freeze-w2v",
            action="store_true",
            help="if we want to freeze the w2v features"
        )

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        return S2TTransformerEncoderW2V2(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)
        # embedding matrix, shared with encoder/decoder/ctc
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder_embed_tokens = decoder_embed_tokens
        # encoder and decoder
        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        # load mt pre-trained models
        mt_pretraining_path = getattr(args, "load_pretrained_mt_encoder_decoder_from", None)
        if mt_pretraining_path is not None and Path(mt_pretraining_path).exists():
            mt_state = checkpoint_utils.load_checkpoint_to_cpu(mt_pretraining_path)
            mt_encoder_state_dict = OrderedDict()
            mt_decoder_state_dict = OrderedDict()
            for key in mt_state["model"].keys():
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    mt_encoder_state_dict[subkey] = mt_state["model"][key]
                if key.startswith("decoder"):
                    subkey = key[len("decoder") + 1 :]
                    mt_decoder_state_dict[subkey] = mt_state["model"][key]
            encoder.load_state_dict(mt_encoder_state_dict, strict=False)
            decoder.load_state_dict(mt_decoder_state_dict, strict=False)
            logger.info(f"loaded pretrained mt encoder and decoder from: {mt_pretraining_path}")
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, audio, audio_lengths, source, source_lengths, prev_output_tokens, align_pad, align_lengths):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(audio, audio_lengths, source, source_lengths, align_pad, align_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class S2TTransformerEncoderW2V2(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, dictionary=None, embed_tokens=None):
        super().__init__(dictionary)

        self.task = args.s2t_task
        self.mixup = getattr(args, "mixup", False)
        self.mixup_arguments = getattr(args, "mixup_arguments", None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0
        self.epoch = 1

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.w2v2_model_path = getattr(args, "w2v2_model_path", None)
        self.freeze_w2v = getattr(args, "freeze_wav", False)
        assert self.w2v2_model_path is not None
        # wav2vec2 model
        ckpt = torchHLoad(self.w2v2_model_path)
        self.w2v_args = ckpt["args"]
        self.wav2vec_model = Wav2Vec2Model.build_model(ckpt["args"], task=None)
        self.wav2vec_model.load_state_dict(ckpt["model"])
        # conv
        w2v_output_dim = self.w2v_args.encoder_embed_dim
        self.subsample = Conv1dSubsampler(
            w2v_output_dim,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        # text encoder
        if self.task == "stack":
            self.embed_tokens = embed_tokens
            export = getattr(args, "export", False)
            if getattr(args, "layernorm_embedding", False):
                self.layernorm_embedding = LayerNorm(embed_tokens.embedding_dim, export=export)
            else:
                self.layernorm_embedding = None
            self.embed_source_positions = PositionalEmbedding(
                args.max_audio_positions,
                args.encoder_embed_dim,
                self.padding_idx
            )
            self.transformer_translation_layers = nn.ModuleList(
                [TransformerEncoderLayer(args) for _ in range(args.translation_encoder_layers)]
            )
            if args.encoder_normalize_before:
                self.translation_layer_norm = LayerNorm(args.encoder_embed_dim)
            else:
                self.translation_layer_norm = None
    
    def _get_wav_features(self, audio, audio_lengths):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(audio_lengths)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(audio, padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return w2v_feature, padding_mask, output_length

    def encode_audio(self, audio, audio_lengths):
        if self.freeze_w2v:
            with torch.no_grad():
                w2v_feature, audio_encoder_padding_mask, input_lengths = self._get_wav_features(audio, audio_lengths)
        else:
            w2v_feature, audio_encoder_padding_mask, input_lengths = self._get_wav_features(audio, audio_lengths)
        audio, input_lengths = self.subsample(w2v_feature, input_lengths)
        audio = self.embed_scale * audio
        audio_encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        audio = self.dropout_module(audio)

        return audio, audio_encoder_padding_mask
    
    def encode_text(self, source, source_encoder_padding_mask):
        for layer in self.transformer_translation_layers:
            source = layer(source, source_encoder_padding_mask)
        if self.translation_layer_norm is not None:
            source = self.translation_layer_norm(source)
        return source, source_encoder_padding_mask
    
    def get_mixed_input(self, audio, source, align_pad, align_lengths, prob):
        mix_output = mix_input(audio, source, align_pad, align_lengths, prob)
        mixseq, mixseq_encoder_padding_mask = mix_output
        positions = self.embed_source_positions(mixseq_encoder_padding_mask).transpose(0, 1)
        mixseq += positions
        if self.layernorm_embedding is not None:
            mixseq = self.layernorm_embedding(mixseq)
        mixseq = self.dropout_module(mixseq)
        return mixseq, mixseq_encoder_padding_mask

    def encode_stack(self, audio, audio_encoder_padding_mask, source, align_pad, align_lengths):
        if not self.mixup:
            return self.encode_text(audio, audio_encoder_padding_mask)
        else:
            source, source_encoder_padding_mask = self.forward_embedding(source)
            prob = get_prob(self.num_updates, self.mixup_arguments, self.training)
            source, source_encoder_padding_mask = self.get_mixed_input(audio, source, align_pad, align_lengths, prob)
            return self.encode_text(source, source_encoder_padding_mask)
            
    def forward_embedding(self, source):
        source_encoder_padding_mask = source.eq(self.padding_idx)
        has_pads = source.device.type == "xla" or source_encoder_padding_mask.any()
        x = embed = self.embed_scale * self.embed_tokens(source)
        if has_pads:
            x = x * (1 - source_encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        return x, source_encoder_padding_mask

    def _forward(self, audio, audio_lengths, source, source_lengths, align_pad, align_lengths):
        audio, audio_encoder_padding_mask = self.encode_audio(audio, audio_lengths)
        if self.task in ("asr", "st"):
            encoder_out, encoder_padding_mask = audio, audio_encoder_padding_mask
        elif self.task == "stack":
            encoder_out, encoder_padding_mask = self.encode_stack(audio, audio_encoder_padding_mask, source, align_pad, align_lengths)

        return save_to_dict(encoder_out, encoder_padding_mask)

    def forward(self, audio, audio_lengths, source, source_lengths, align_pad, align_lengths):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(audio, audio_lengths, source, source_lengths, align_pad, align_lengths)
        else:
            x = self._forward(audio, audio_lengths, source, source_lengths, align_pad, align_lengths)
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
    
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch


@register_model_architecture(model_name="s2t_transformer_w2v2", arch_name="s2t_transformer_b_w2v_0tenc_6dec")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # wav2vec2
    args.w2v2_model_path = getattr(args, "w2v2_model_path", "./wav2vec_small_100h.pt")
    args.freeze_w2v = getattr(args, "freeze_w2v", False)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.translation_encoder_layers = getattr(args, "translation_encoder_layers", 0)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("s2t_transformer_w2v2", "s2t_transformer_s_w2v_0tenc_6dec")
def s2t_transformer_s_12aenc_0tenc_6dec(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_transformer_w2v2", "s2t_transformer_s_w2v_6tenc_6dec")
def s2t_transformer_s_12aenc_6tenc_6dec(args):
    args.translation_encoder_layers = getattr(args, "translation_encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_transformer_w2v2", "s2t_transformer_b_w2v_6tenc_6dec")
def s2t_transformer_b_12aenc_6tenc_6dec(args):
    args.translation_encoder_layers = getattr(args, "translation_encoder_layers", 6)
    base_architecture(args)
