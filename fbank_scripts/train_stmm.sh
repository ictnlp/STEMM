lang=$1
exp=mustc_en${lang}_stmm_self_learning_fbank
fairseq-train data/mustc/en-${lang} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --s2t-task stack --criterion label_smoothed_cross_entropy_with_stmm_self_learning --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s_12aenc_6tenc_6dec --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 2 \
  --patience 10 \
  --max-epoch 25 \
  --load-pretrained-asr-encoder-from checkpoints/mustc_en${lang}_asr.pt \
  --load-pretrained-mt-encoder-decoder-from checkpoints/mustc_en${lang}_mt.pt \
  --mixup --mixup-arguments fix,100000,1.0 | tee logs/${exp}.txt