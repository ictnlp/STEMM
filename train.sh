lang=$1
exp=mustc_en${lang}_stmm_self_learning
fairseq-train data/mustc/en-${lang} \
  --config-yaml config_raw.yaml --train-subset train_raw --valid-subset dev_raw \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 2000000 --max-update 100000 \
  --task speech_to_text --s2t-task stack --criterion label_smoothed_cross_entropy_with_stmm_self_learning --label-smoothing 0.1 \
  --arch s2t_transformer_b_w2v_6tenc_6dec --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 1 \
  --layernorm-embedding \
  --patience 10 \
  --max-epoch 25 \
  --max-audio-positions 900000 \
  --fp16 \
  --w2v2-model-path checkpoints/wav2vec_small.pt \
  --load-pretrained-mt-encoder-decoder-from checkpoints/mustc_en${lang}_mt.pt \
  --mixup --mixup-arguments fix,100000,1.0 | tee logs/${exp}.txt