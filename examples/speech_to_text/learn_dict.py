import os
import argparse
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile
from examples.speech_to_text.data_utils import gen_vocab, load_df_from_tsv, gen_config_yaml
from examples.speech_to_text.prep_mustc_data import MUSTC

split = "train"

def process(args):
    root = Path(args.data_root).absolute()
    for lang in MUSTC.LANGUAGES:
        cur_root = root / f"en-{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        df = load_df_from_tsv(cur_root / f"{split}_seg.tsv")
        train_text = []
        for _, row in df.iterrows():
            train_text.append(row["src_text"])
            train_text.append(row["tgt_text"])
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        gen_config_yaml(
            cur_root,
            spm_filename_prefix + ".model",
            yaml_filename=f"config.yaml",
            specaugment_policy="lb",
            cmvn_type=args.cmvn_type
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()