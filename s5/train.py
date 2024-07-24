from functools import partial
import jax
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
from flax.training import checkpoints
import wandb

from .train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from .dataloading import Datasets
from .seq_model import BatchClassificationModel, RetrievalModel
from .ssm import init_S5SSM
from .ssm_init import make_DPLR_HiPPO

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, 
                   name=f"{args.dataset}_seed{args.jax_seed}_ep_{args.epochs}_pep_{args.pruning_epoch}_{args.pruning_method}",
                   tags=[args.dataset],
                   config=vars(args), entity=args.wandb_entity)
    else:
        wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Dataset dependent logic
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False

    else:
        padded = False
        retrieval = False

    # For speech dataset
    if args.dataset in ["speech35-classification"]:
        speech = True
        print("Will evaluate on both resolutions for speech task")
    else:
        speech = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
      create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting S5 Training on `{args.dataset}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    if args.pruning_method in ["uniformHinf", "globalHinf"]:
        criterion = "Hinf"
    elif args.pruning_method in ["random", "LAST"]:
        criterion = "LAST"

    ssm_init_fn = init_S5SSM(H=args.d_model,
                             P=ssm_size,
                             Lambda_re_init=Lambda.real,
                             Lambda_im_init=Lambda.imag,
                             V=V,
                             Vinv=Vinv,
                             C_init=args.C_init,
                             discretization=args.discretization,
                             dt_min=args.dt_min,
                             dt_max=args.dt_max,
                             criterion=criterion,
                             conj_sym=args.conj_sym,
                             clip_eigs=args.clip_eigs,
                             bidirectional=args.bidirectional,
                             pruning=args.pruning)

    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    if args.pruning:
        total_remaining_dim = max(int(ssm_size * args.n_layers * (100 - args.pruning_ratio) / 100), 1) # clip for ratio = 100
        history_dir = f"results/{args.dataset}/{args.jax_seed}/"
        os.makedirs(history_dir, exist_ok=True)
        history_fname = f"{history_dir}/eps_{args.epochs}_pep_{args.pruning_epoch}_{args.pruning_method}.npz"
        plot_fname = f"{history_dir}/eps_{args.epochs}_pep_{args.pruning_epoch}_{args.pruning_method}.png"
        if os.path.isfile(history_fname):
            npzfile = np.load(history_fname)
            score_history = npzfile['score_history']
            test_acc_history = npzfile['test_acc_history']
            if 'score_for_mask' in npzfile.keys():
                score_for_mask = npzfile['score_for_mask']
            else:
                score_for_mask = score_history[args.pruning_epoch]
            th = np.sort(np.concatenate(score_history[args.pruning_epoch+1]))[-total_remaining_dim]
        
        else:
            # initialize training state
            state = create_train_state(model_cls,
                                    init_rng,
                                    padded,
                                    retrieval,
                                    in_dim=in_dim,
                                    bsz=args.bsz,
                                    seq_len=seq_len,
                                    weight_decay=args.weight_decay,
                                    batchnorm=args.batchnorm,
                                    opt_config=args.opt_config,
                                    ssm_lr=ssm_lr,
                                    lr=lr,
                                    dt_global=args.dt_global)

            # Training Loop over epochs

            th = 0
            score_for_mask = [None for _ in range(args.n_layers)]
            score_history = []
            test_acc_history = []
            if valloader is not None:
                _, _, score = validate(state,
                                            model_cls,
                                            valloader,
                                            seq_len,
                                            in_dim,
                                            args.batchnorm,
                                            th=th,
                                            score_for_mask=score_for_mask)
                _, test_acc, _ = validate(state,
                                                model_cls,
                                                testloader,
                                                seq_len,
                                                in_dim,
                                                args.batchnorm,
                                                th=th,
                                                score_for_mask=score_for_mask)
                test_acc_history.append(test_acc)
            else:
                _, val_acc, score = validate(state,
                                            model_cls,
                                            testloader,
                                            seq_len,
                                            in_dim,
                                            args.batchnorm)
                test_acc_history.append(val_acc)
            score_history.append(score)
            
            best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
            count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
            lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
            step = 0  # for per step learning rate decay
            steps_per_epoch = int(train_size/args.bsz)
            for epoch in range(args.epochs):
                if epoch > args.pruning_epoch:
                    print(f"[*] Starting Training Epoch {epoch + 1} with Pruned Model...")
                else:
                    print(f"[*] Starting Training Epoch {epoch + 1}...")

                if epoch < args.warmup_end:
                    print("using linear warmup for epoch {}".format(epoch+1))
                    decay_function = linear_warmup
                    end_step = steps_per_epoch * args.warmup_end

                elif args.cosine_anneal:
                    print("using cosine annealing for epoch {}".format(epoch+1))
                    decay_function = cosine_annealing
                    # for per step learning rate decay
                    end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
                else:
                    print("using constant lr for epoch {}".format(epoch+1))
                    decay_function = constant_lr
                    end_step = None

                # TODO: Switch to letting Optax handle this.
                #  Passing this around to manually handle per step learning rate decay.
                lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

                train_rng, skey = random.split(train_rng)
                state, train_loss, step = train_epoch(state,
                                                    skey,
                                                    model_cls,
                                                    trainloader,
                                                    seq_len,
                                                    in_dim,
                                                    args.batchnorm,
                                                    lr_params,
                                                    th=th,
                                                    score_for_mask=score_for_mask)

                if valloader is not None:
                    print(f"[*] Running Epoch {epoch + 1} Validation...")
                    val_loss, val_acc, score = validate(state,
                                                model_cls,
                                                valloader,
                                                seq_len,
                                                in_dim,
                                                args.batchnorm,
                                                th=th,
                                                score_for_mask=score_for_mask)

                    print(f"[*] Running Epoch {epoch + 1} Test...")
                    test_loss, test_acc, _ = validate(state,
                                                model_cls,
                                                testloader,
                                                seq_len,
                                                in_dim,
                                                args.batchnorm,
                                                th=th,
                                                score_for_mask=score_for_mask)

                    print(f"\n=>> Epoch {epoch + 1} Metrics ===")
                    print(
                        f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                        f" Val Accuracy: {val_acc:.4f}"
                        f" Test Accuracy: {test_acc:.4f}"
                    )
                    test_acc_history.append(test_acc)

                else:
                    # else use test set as validation set (e.g. IMDB)
                    print(f"[*] Running Epoch {epoch + 1} Test...")
                    val_loss, val_acc, score = validate(state,
                                                model_cls,
                                                testloader,
                                                seq_len,
                                                in_dim,
                                                args.batchnorm,
                                                th=th,
                                                score_for_mask=score_for_mask)

                    print(f"\n=>> Epoch {epoch + 1} Metrics ===")
                    print(
                        f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                        f" Test Accuracy: {val_acc:.4f}"
                    )
                    test_acc_history.append(val_acc)

                score_history.append(score)
                # Pruning
                if (epoch + 1) == args.pruning_epoch:
                    print(f"\n=>> Epoch {epoch + 1} Pruning ===")
                    if args.pruning_method in ["globalHinf", "LAST"]:
                        th = np.sort(score.reshape(-1))[-total_remaining_dim]
                        score_for_mask = score
                    elif args.pruning_method == "random":
                        th = args.pruning_ratio / 100
                        init_rng, key = random.split(init_rng, num=2)
                        score_for_mask = random.uniform(key, score.shape)
                    elif args.pruning_method == "uniformHinf":
                        score_for_mask = score
                        pass #TODO: local threshold

                    print(f"Global threshold for scores (Pruning ratio: {args.pruning_ratio}%): {th:.5f}")
                    
                    print(f"[*] Evaluating pruning...")
                    test_loss, test_acc, _ = validate(state,
                                                    model_cls,
                                                    testloader,
                                                    seq_len,
                                                    in_dim,
                                                    args.batchnorm,
                                                    th=th,
                                                    score_for_mask=score)

                    print(f"\tTest Accuracy: {test_acc:.4f}")


                # For early stopping purposes
                if val_loss < best_val_loss:
                    count = 0
                    best_val_loss = val_loss
                else:
                    count += 1

                if val_acc > best_acc:
                    # Increment counters etc.
                    count = 0
                    best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
                    if valloader is not None:
                        best_test_loss, best_test_acc = test_loss, test_acc
                    else:
                        best_test_loss, best_test_acc = best_loss, best_acc

                    # Do some validation on improvement.
                    if speech:
                        # Evaluate on resolution 2 val and test sets
                        print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                        val2_loss, val2_acc = validate(state,
                                                    model_cls,
                                                    aux_dataloaders['valloader2'],
                                                    int(seq_len // 2),
                                                    in_dim,
                                                    args.batchnorm,
                                                    step_rescale=2.0,
                                                    th=th,
                                                    score_for_mask=score_for_mask)

                        print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                        test2_loss, test2_acc = validate(state, model_cls, aux_dataloaders['testloader2'], int(seq_len // 2), in_dim, args.batchnorm, step_rescale=2.0)
                        print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                        print(
                            f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                            f" Val Accuracy: {val2_acc:.4f}"
                            f" Test Accuracy: {test2_acc:.4f}"
                        )

                # For learning rate decay purposes:
                input = lr, ssm_lr, lr_count, val_acc, opt_acc
                lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

                # Print best accuracy & loss so far...
                print(
                    f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
                    f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
                    f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
                    f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
                )

                if valloader is not None:
                    if speech:
                        wandb.log(
                            {
                                "Training Loss": train_loss,
                                "Val loss": val_loss,
                                "Val Accuracy": val_acc,
                                "Test Loss": test_loss,
                                "Test Accuracy": test_acc,
                                "Val2 loss": val2_loss,
                                "Val2 Accuracy": val2_acc,
                                "Test2 Loss": test2_loss,
                                "Test2 Accuracy": test2_acc,
                                "count": count,
                                "Learning rate count": lr_count,
                                "Opt acc": opt_acc,
                                "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                                "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                            }
                        )
                    else:
                        wandb.log(
                            {
                                "Training Loss": train_loss,
                                "Val loss": val_loss,
                                "Val Accuracy": val_acc,
                                "Test Loss": test_loss,
                                "Test Accuracy": test_acc,
                                "count": count,
                                "Learning rate count": lr_count,
                                "Opt acc": opt_acc,
                                "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                                "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                            }
                        )

                else:
                    wandb.log(
                        {
                            "Training Loss": train_loss,
                            "Val loss": val_loss,
                            "Val Accuracy": val_acc,
                            "count": count,
                            "Learning rate count": lr_count,
                            "Opt acc": opt_acc,
                            "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                            "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                        }
                    )
                wandb.run.summary["Best Val Loss"] = best_loss
                wandb.run.summary["Best Val Accuracy"] = best_acc
                wandb.run.summary["Best Epoch"] = best_epoch
                wandb.run.summary["Best Test Loss"] = best_test_loss
                wandb.run.summary["Best Test Accuracy"] = best_test_acc

                if count > args.early_stop_patience:
                    break
            test_acc_history = np.array(test_acc_history)
            score_history = np.array(score_history)
            np.savez(history_fname, 
                     score_history=score_history, 
                     test_acc_history=test_acc_history,
                     score_for_mask=score_for_mask)

    # Prune and test
    if args.pruning:
        score_history = np.transpose(score_history, (1,0,2)) # [E, L, (P/2)] -> [L, E, (P/2)]
        
        colors = cm.viridis(np.linspace(0, 1, ssm_size))
        nrows = 1 + (args.n_layers+1)//2
        height_ratios = [1.8] + [1 for _ in range(nrows-1)]
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(9, 2+1.3*nrows), gridspec_kw={'height_ratios': height_ratios})
        
        xticklabels = [str(i) for i in range(1, args.epochs+1)]
        xticklabels = ["Init"] + xticklabels

        # (1) Test accuracy history
        ax = axes[0, 0]
        ax.plot(np.arange(args.epochs+1), test_acc_history, color='mediumpurple', lw=2.5, marker='o', markersize=7)
        ax.axvline(x=args.pruning_epoch, color="olive", lw=2, linestyle='--',zorder=100)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test accuracy')
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_xticks(np.arange(args.epochs + 1))
        ax.set_xticklabels(xticklabels)
        # ax.set_ylim([0.85, 1.0])

        # (2) Profiling history
        ax_latency = axes[0, 1]
        ax_memory = ax_latency.twinx()
        # ------임시----- #
        line1, = ax_latency.plot(np.arange(args.epochs+1), test_acc_history, color='sandybrown', lw=2.5, marker='D', markersize=7, label='Latency')
        line2, = ax_memory.plot(np.arange(args.epochs+1), score_history[0, :, 0], color='orchid', lw=2.5, marker='v', markersize=7, label='Memory')
        # ------임시----- #
        ax_latency.axvline(x=args.pruning_epoch, color="olive", lw=2, linestyle='--',zorder=100)
        ax_latency.set_xlabel('Epoch')
        ax_latency.set_ylabel('Latency (ms)')
        ax_memory.set_ylabel('Memory usage (MB)')
        ax_latency.yaxis.grid(True, alpha=0.4)
        ax_latency.set_xticks(np.arange(args.epochs + 1))
        ax_latency.set_xticklabels(xticklabels)
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax_latency.legend(lines, labels, loc='lower right')

        # (3) Score history
        for l in range(args.n_layers):
            ax = axes[1+l // 2, l % 2]
            initial_scores = score_history[l, 0, :]  # based on the initial value
            sorted_indices = np.argsort(initial_scores)
            color_map = {idx: colors[sorted_indices.tolist().index(idx)] for idx in range(ssm_size)}
            for s in range(ssm_size):
                if score_for_mask[l,s] < th:
                    ax.plot(np.arange(0, args.pruning_epoch+1), score_history[l,:args.pruning_epoch+1,s], color=color_map[s])
                    ax.plot(np.arange(args.pruning_epoch, args.epochs+1), score_history[l,args.pruning_epoch:,s], color="silver")
                else:
                    ax.plot(np.arange(args.epochs+1), score_history[l,:,s], color=color_map[s])

            # current pruning
            # ax.axhline(y=th, color="olive", lw=2, linestyle='--',zorder=100)
            ax.axvline(x=args.pruning_epoch, color="olive", lw=2, linestyle='--',zorder=100)
            ax.set_yscale('log')
            ax.set_title(f'Layer {l+1}')
            ax.set_xlabel('Epoch')
            if criterion == 'LAST':
                ax.set_ylabel('LAST score')
            elif criterion == 'Hinf':
                ax.set_ylabel(r'$\mathcal{H}_{\infty}$ score')

            ax.set_xticks(np.arange(args.epochs + 1))
            ax.set_xticklabels(xticklabels)

        plt.tight_layout()
        plt.savefig(plot_fname)
