lang=$1
exp=mustc_en${lang}_mt
fairseq-train data/mustc/en-${lang} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 4096 --max-update 100000 \
  --task speech_to_text --s2t-task mt --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s_0aenc_6tenc_6dec --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 8000 --clip-norm 10.0 --seed 1 --update-freq 2 \
  --layernorm-embedding \
  --patience 10 | tee logs/${exp}.txt