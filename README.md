# STEMM: Self-learning with **S**peech-**T**ext **M**anifold **M**ixup for Speech Translation

This is a PyTorch implementation for the ACL 2022 main conference paper [STEMM: Self-learning with Speech-text Manifold Mixup for Speech Translation](https://arxiv.org/abs/2203.10426).

## Training a Model on MuST-C

Let's first take a look at training an En-De model as an example.

### Enviroment Configuration

1. Clone this repository:

```shell
git clone git@github.com:ictnlp/STEMM.git
cd STEMM/
```

2. Install Montreal Forced Aligner following the [official guidance](https://montreal-forced-aligner.readthedocs.io/en/v1.0/installation.html). Please also download the pertained models and dictionary for MFA.

3. Please make sure you have installed PyTorch, and then install fairseq and other packages as follows:

```shell
pip install --editable ./
python3 setup.py install --user
python3 setup.py build_ext --inplace
pip install inflect sentencepiece soundfile textgrid pandas
```

### Data Preparation

1. First make a directory to store the dataset:

```shell
TGT_LANG=de
MUSTC_ROOT=data/mustc/
mkdir -p $MUSTC_ROOT
```

2. Download the [MuST-C v1.0](https://ict.fbk.eu/must-c/) archive `MUSTC_v1.0_en-de.tar.gz` to the `$MUSTC_ROOT` path, and uncompress it:

```shell
cd $MUSTC_ROOT
tar -xzvf MUSTC_v1.0_en-de.tar.gz
```

3. Return to the root directory, run the preprocess script `preprocess.sh`, which will perform forced alignment and organize the raw data and alignment information into `.tsv` format for using:

```shell
sh preprocess.sh $TGT_LANG
```

4. Finally, the directory `$MUSTC_ROOT` should look like this:

```
.
├── en-de
│   ├── config_raw.yaml
│   ├── data
│   ├── dev_raw_seg_plus.tsv
│   ├── docs
│   ├── segment
│   ├── spm_unigram10000_raw.model
│   ├── spm_unigram10000_raw.txt
│   ├── spm_unigram10000_raw.vocab
│   ├── train_raw_seg_plus.tsv
│   ├── tst-COMMON_raw_seg_plus.tsv
│   ├── tst-HE_raw_seg_plus.tsv
└── MUSTC_v1.0_en-de.tar.gz
```

### Pretrain the MT Module

#### [OPTIONAL] Use External MT Corpus

If you want to use external MT corpus, please first pretrain a MT model on this corpus following these steps:

1. Perform BPE on external corpus with the sentencepiece model learned on MuST-C. As we mentioned in our paper, we use WMT for En-De, En-Fr, En-Ru, En-Es, En-Ro, and OPUS100 for En-Pt, En-It, En-Nl as external corpus. You can download them from the internet and put them in the `data/ext_en${TGT_LANG}/` directory. Run the following command and replace `$input_file` with the path of raw text to perform BPE. You should apply BPE to texts in both source and target language of all subset (train/valid/test).

```shell
python3 data/scripts/apply_spm.py --input-file $input_file --output-file $output_file --model data/mustc/en-${TGT_LANG}/spm_unigram10000_raw.model
```

2. Use `fairseq-preprocess` command to convert the BPE texts into fairseq formats. Make sure to use the sentencepiece dictionary learned on MuST-C.

```shell
$spm_dict=data/mustc/en-${TGT_LANG}/spm_unigram10000_raw.txt
fairseq-preprocess --source-lang en --target-lang $TGT_LANG --trainpref data/ext_en${TGT_LANG}/train --validpref data/ext_en${TGT_LANG}/valid --testpref data/ext_en${TGT_LANG}/test --destdir data/ext_en${TGT_LANG}/binary --joined-dictionary --srcdict $spm_dict --tgtdict $spm_dict --workers=20 --nwordssrc 10000 --nwordstgt 10000
```

3. Train the model using the following command:

```shell
sh pretrain_mt_ext.sh $TGT_LANG
```

#### Pretrain the MT module on MuST-C

1. Run the following script to pretrain the MT module. The argument `--load-pretrained-mt-encoder-decoder-from` indicates the path of MT model pretrained on external corpus obtained in the last step.

```shell
sh pretrain_mt.sh $TGT_LANG
```

2. **To ensure consistent performance, we have released our checkpoints of pretrained MT modules**. You can download them and directly use them do initialize the MT module in our model for the following experiments.

| Direction | Link |
| --------- | ---- |
| En-De     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_ende_mt.pt    |
| En-Fr     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enfr_mt.pt    |
| En-Es     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enes_mt.pt    |
| En-Ro     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enro_mt.pt    |
| En-Ru     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enru_mt.pt    |
| En-Nl     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_ennl_mt.pt    |
| En-It     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enit_mt.pt    |
| En-Pt     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enpt_mt.pt    |

### Training

1. Download the pretrained wav2vec2.0 model from the [official link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt), and put it in the `checkpoints/` directory.
2. Just run the training scripts:

```shell
sh train.sh $TGT_LANG
```

### Evaluate

1. Run the following script to average the last 10 checkpoints and evaluate on the `tst-COMMON` set:

```shell
sh test.sh mustc_en${TGT_LANG}_stmm_self_learning $TGT_LANG
```

2. We also released our checkpoints as follows. You can download and evaluate them directly.

| Direction | Link |
| --------- | ---- |
| En-De     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_ende_stmm_self_learning.pt    |
| En-Fr     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enfr_stmm_self_learning.pt    |
| En-Es     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enes_stmm_self_learning.pt    |
| En-Ro     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enro_stmm_self_learning.pt    |
| En-Ru     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enru_stmm_self_learning.pt    |
| En-Nl     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_ennl_stmm_self_learning.pt    |
| En-It     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enit_stmm_self_learning.pt    |
| En-Pt     |  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/stmm/mustc_enpt_stmm_self_learning.pt    |

## Citation
In this repository is useful for you, please cite as:

```
@inproceedings{fang-etal-2022-STEMM,
	title = {STEMM: Self-learning with Speech-text Manifold Mixup for Speech Translation},
	author = {Fang, Qingkai and Ye, Rong and Li, Lei and Feng, Yang and Wang, Mingxuan},
	booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
	year = {2022},
}
```

## Contact

If you have any questions, feel free to contact me at `fangqingkai21b@ict.ac.cn`.

