lang=$1
exp=mustc_en${lang}_mt
fairseq-train data/mustc/en-${lang} \
  --config-yaml config_raw.yaml --train-subset train_raw --valid-subset dev_raw \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 4096 --max-update 100000 \
  --task speech_to_text --s2t-task mt --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_b_0aenc_6tenc_6dec --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 8000 --clip-norm 10.0 --seed 1 --update-freq 1 \
  --layernorm-embedding \
  --patience 10 \
  --load-pretrained-mt-encoder-decoder-from checkpoints/en${lang}_mt.pt | tee logs/${exp}.txt