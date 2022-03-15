exp=$1
lang=$2
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python3 scripts/average_checkpoints.py \
  --inputs checkpoints/${exp} --num-epoch-checkpoints 10 \
  --output "checkpoints/${exp}/${CHECKPOINT_FILENAME}"
fairseq-generate data/mustc/en-${lang} \
  --config-yaml config_raw.yaml --gen-subset tst-COMMON_raw --task speech_to_text --s2t-task stack \
  --path checkpoints/${exp}/${CHECKPOINT_FILENAME} \
  --max-audio-positions 900000 \
  --max-tokens 2000000 --beam 5 --scoring sacrebleu | tee result/${exp}.txt