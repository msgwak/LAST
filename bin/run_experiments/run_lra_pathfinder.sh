python run_train.py --pruning=True --pruning_ratio=30 \
                    --pruning_epoch=5 --epochs=200 \
                    --C_init=trunc_standard_normal --batchnorm=True --bidirectional=True \
                    --blocks=8 --bn_momentum=0.9 --bsz=64 --d_model=192 \
                    --dataset=pathfinder-classification --jax_seed=8180844 --lr_factor=5 \
                    --n_layers=6 --opt_config=standard --p_dropout=0.05 --ssm_lr_base=0.0009 \
                    --ssm_size_base=256 --warmup_end=1 --weight_decay=0.03 \
                    --USE_WANDB=True --wandb_project=s5