#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import logging
from pathlib import Path
import shutil
import soundfile as sf
from tempfile import NamedTemporaryFile

import pandas as pd
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_config_yaml_raw,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS = [
    "dev-clean",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = LIBRISPEECH(out_root.as_posix(), url=split)
        for wav, sample_rate, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            manifest["id"].append(sample_id)
            manifest["src_text"].append(utt.lower())
            manifest["tgt_text"].append("")
            manifest["speaker"].append(spk_id)
            utt_no_pad = "%04d" % utt_no
            wavpath = os.path.join(out_root, "LibriSpeech_MFA", split, str(spk_id), f"{spk_id}-{chapter_no}-{utt_no_pad}.wav")
            sr = sf.info(wavpath).samplerate
            duration = sf.info(wavpath).duration
            offset = 0
            n_frames = int(duration * sr)
            manifest["audio"].append(f"{wavpath}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}_raw.tsv"
        )
    # Generate config YAML
    gen_config_yaml_raw(
        out_root, 
        None,
        yaml_filename=f"config_raw.yaml"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
