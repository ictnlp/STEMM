lang=$1
exp=en${lang}_mt
fairseq-train data/ext_en${lang}/binary --task translation \
    --arch transformer_wmt_en_de --share-all-embeddings --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --stop-min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 4096 \
    --update-freq 1 --no-progress-bar --log-format json --log-interval 100 \
    --layernorm-embedding \
    --keep-last-epochs 20 \
    --save-dir checkpoints/$exp \
    --ddp-backend=no_c10d \
    --max-update 250000 | tee logs/$exp.txt