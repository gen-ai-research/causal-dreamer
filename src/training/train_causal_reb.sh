export TOKENIZERS_PARALLELISM=false
python -m src.training.train_causal_rebalanced \
  --full_csv ../data/processed/causal_full.csv \
  --train_list ../data/processed/train_list.txt \
  --val_list   ../data/processed/val_list.txt \
  --list_is_indices --list_index_base 0 \
  --batch_size 64 --epochs 200 --lr 2e-4 \
  --lambda_triplet 0.5 --lambda_delta 1.0 --lambda_retr 10.0 \
  --tau 0.07 --symmetric_infonce \
  --save_dir outputs_rebalanced \
  --learnable_logit_scale --logit_scale_init 0.07 --logit_scale_min 0.01 --logit_scale_max 2.0 \
  --tau 0.05 \
  --retr_topk_neg 64 \
  --sched cosine --warmup_epochs 5 --min_lr 1e-6 \
  --symmetric_infonce \
  --ema --ema_decay 0.999 \
  --grad_clip_norm 1.0 \
  --text_max_len 77

