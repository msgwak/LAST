# LAST: Layer-adaptive $\mathcal{H}_{\infty}$ state pruning
### Dear reviewers,

Thank you for your attention to our work.

Our method is built on the top of the [S5](https://github.com/lindermanlab/S5) and [Annotated S4](https://github.com/srush/annotated-s4) repositories.
Specifically, our repository provides the code for training and pruning S5 models.
Our modified (pruning version) `S5SSM` class evaluates `LASTscore`, which are used to generate corresponding system masks.
The evaluation of LAST scores matches the final score equation in the paper, as impleneted in `ssm.py`.

For the environment setup, we have provided our current environment along with the original environment from S5.
Please ensure that you use the correct versions of JAX-associated libraries.

Sincerely,

LAST authors.

![](./docs/figures/pngs/last.png)
<p style="text-align: center;">
Figure 1:  Illustration of the LAST for two layers. By normalizing the layer-level energy loss by the total output energy when the states with lower H_8 scores are excluded, we measure the model-level energy loss (LAST score) and prune the states with low LAST scores. 
</p>

---

### Dataset downloading (Provided by [S5](https://github.com/lindermanlab/S5))

- For LRA dataset
    - `.bin/download_lra.sh`

### Experiments
- `.bin/run_experiments/run_lra_listops.sh`
- `.bin/run_experiments/run_lra_imdb.sh`
- `.bin/run_experiments/run_lra_aan.sh`
- `.bin/run_experiments/run_lra_cifar.sh`
- `.bin/run_experiments/run_lra_pathfinder.sh`
- `.bin/run_experiments/run_lra_pathx.sh`