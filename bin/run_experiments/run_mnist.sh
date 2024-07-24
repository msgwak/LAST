python run_train.py --pruning=True --pruning_ratio=50 \
                    --pruning_method=globalHinf \
                    --pruning_epoch=1 --epochs=5 \
                    --dataset=mnist-classification \
                    --n_layers=4 --d_model=96 --ssm_size_base=128 --blocks=1 \
                    --p_dropout=0.1 --lr_factor=4 --ssm_lr_base=0.002 \
                    --bsz=50 --weight_decay=0.01 \
                    --jax_seed=2024 \
                    # --USE_WANDB=True --wandb_project=s5