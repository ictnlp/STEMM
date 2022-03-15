lang=$1
exp=mustc_en${lang}_asr_base
fairseq-train data/mustc/en-${lang} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --s2t-task asr --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_b_12aenc_0tenc_6dec --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 1 \
  --patience 10 | tee logs/${exp}.txt