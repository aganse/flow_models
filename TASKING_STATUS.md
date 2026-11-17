# TASKING: flow_models use cases 3-7

**STATUS: all six stages complete.  This is the final version, written as next
steps for the repo owner.**

Unzip over the root of a checkout of `aganse/flow_models` (built against commit
`e3f3e5a`).  Nothing has been committed anywhere.

**The one thing to know first: none of the TensorFlow code has been run.**
The environment this was written in had no TensorFlow, TFP or MLflow and could
not install them.  Section 3 says exactly what was and was not checked; section
2 is the order to run things in to find problems quickly.


## 1. What is in the zip

### New files
| File | Purpose |
|---|---|
| `train_flowmodels3.py`, `params/params3.json` | Use case 3: cat/dog generative classification (Glow, likelihood-trained). Laid out like script 2. |
| `train_flowmodels4.py`, `params/params4.json` | Use case 4: 8-mode GMM with 4 color labels (Ardizzone Fig 2). Laid out like script 1. |
| `train_flowmodels5.py`, `params/params5.json` | Use case 5: inverse kinematics (Ardizzone Fig 3) with a rejection-ABC reference posterior. |
| `train_flowmodels6.py`, `params/params6.json` | Use case 6: glacier gravimetry, noise-free (Zhang & Curtis). Holds `run_glacier_inversion()`. |
| `train_flowmodels7.py`, `params/params7.json` | Use case 7: same with noise as a network input. Own parameter dicts; imports `run_glacier_inversion` from script 6. |
| `forward_models.py` | numpy-only physics and priors: `InvKinProblem`, `gravprob` (port of `gravprob.m`), `GlacGravProblem`. |
| `inn_utils.py` | numpy/matplotlib metrics and plots shared by scripts 3-7. |
| `tests/test_inn_mode.py` | Unit tests for the new FlowModel mode and labeled generators (needs TensorFlow; **not yet run**). |
| `tests/test_forward_models.py` | Unit tests for the forward models and priors (numpy only; these pass). |
| `TASKING.md` | This file. Not meant to stay in the repo. |

### Modified files
| File | Change |
|---|---|
| `flow_model.py` | New "INN mode", on when `y_dim > 0`: constructor args `y_dim`, `z_dim`, `x_dim`, `lambda_y`, `lambda_z`, `lambda_x`, `lambda_pad`, `lambda_nll`, `y_sigma`, `pad_noise_scale`, `mmd_bandwidths` (all saved by `get_config`); `train_step` hands off to `_inn_train_step`; new methods `_data_to_latent`, `split_latent`, `_mmd`, `_inn_losses`, `pad_inputs`, `forward_yz`, `sample_posterior`. With `y_dim` 0 (the default) the original code path is untouched. Additions only; no existing lines removed. |
| `model_training.py` | `_make_dataset` only: with `y_dim > 0` it yields (x, y) pairs instead of dropping y. |
| `file_utils.py` | `get_data_generator`: optional `pad_to`, `classes`, `shuffle`, `problem`, `dataset_params`; `class_mode="categorical"` returns labels for `gmm` and `catsdogs`; `invkin` and `glacgrav` implemented. `S3ImageDataGenerator.flow_from_directory`: `classes` arg and `sparse`/`categorical` labels. `dequantize_generator` passes labels through. New `pad_to_dim`, `GMM_LABEL_MAP`, `GMM_LABEL_COLORS`. Calls made the way scripts 1-2 make them behave as before. |
| `aws/job-support-common/makefile-common.mk`, `aws/job-support-sagemaker/makefile-sagemaker.mk` | The "IMAGES_PATH required" guards apply to scripts 2 and 3 only (were: every script but 1). |
| `doc/config.md` | Every new parameter, App# columns extended to 1-7, new INN-mode section. |

Not touched: `Dockerfile` (`COPY *.py` already picks up the new modules),
`requirements.txt` (no new packages), `makefile`, `utils.py`, `README.md`.


## 2. What to do next, in order

1. **Run the test suite** in the repo's environment: `make test`.
   - The pre-existing tests should pass unchanged. If any fail, the default
     (`y_dim` 0) path was disturbed, which should not happen.
   - `tests/test_inn_mode.py` is the first real execution of the new code.
     Failures there point straight at the item to fix. The most informative one
     is `test_nll_matches_shifted_base_density`, which checks the likelihood
     term against `FlowModel.log_prob`.
2. **Script 4 for 1 epoch** (`num_epochs` 1, CPU is fine). It is the smallest
   problem and exercises padding, all MMD terms and the full MLflow logging.
   Check that `loss_y`, `loss_z`, `loss_x`, `loss_pad` all appear in MLflow.
3. **Script 4 for real**, then look at:
   - `simpts_dataspace.png`: each color should reproduce its own clusters
     (compare Ardizzone Fig 2). Red must split across four clusters.
   - `trainpts_latentspace.png`: all four colors overlapping in one Gaussian
     blob inside the dashed 50%/90% circles.
   - `label_accuracy` near 1.
4. **Script 5**: `arm_posterior_target1.png` should show the INN panel matching
   the ABC panel (compare Ardizzone Fig 3); `targetK_mmd_inn_vs_abc` small;
   `calibration_err_median` of a few percent or less.
5. **Script 6**, default prior, then script 7. Look at:
   - `forward_rmse_mgal`: how well the network learned the physics. It should
     be well under 1 mGal; if not, nothing downstream is meaningful.
   - `depth_posterior_synthetic.png`: the true sin^2 profile should sit inside
     the posterior band, and the band should be narrower than the dotted prior
     band under the stations.
   - `data_fit_*.png`: predicted data through the observed points.
   - Script 7's `depth_std_*.png` should be wider than script 6's.
   - Then rerun with `prior_type` `"smooth_gaussian_hierarchical"` and
     `"uniform"` for the comparison (see item 6 of section 4 about uniform).
6. **Script 3** last, on a GPU. It costs about as much as script 2.
   - Needs `IMAGES_PATH/train/cat`, `train/dog`, `val/cat`, `val/dog`.
   - Watch `val_accuracy` against `train_accuracy`, and
     `val_points_labelspace.png` (two clusters at the black crosses).
7. If you keep the work: add scripts 3-7 to the README's run instructions
   (they still say "N is 1, 2"), and delete this file.


## 3. What was and was not verified

| Item | Status |
|---|---|
| `forward_models.py` | **Run.** `gravprob` agrees with a fine-grid quadrature of Parker eqn 5.02(1) to about 0.02 mGal. Priors, noise handling, standardization, inverse kinematics and ABC all exercised; 12 unit tests pass. |
| `inn_utils.py` | **Run** on synthetic data, every function. |
| Scripts 3-7, their own logic | **Run with a stand-in model.** Each `main()` was executed end to end with TensorFlow/MLflow stubbed and the trained model replaced by an object returning random arrays of the right shapes. That covers the real data generators, parameter and JSON plumbing, (x, y) batch shapes and dtypes, every metric and every plot. Scripts 6 and 7 were run with all three prior types. Script 3 was run on generated dummy images through the local-directory branch of `S3ImageDataGenerator`. |
| `params/params3-7.json` | Generated from each script's default dicts; every `model_arch_params` key checked against the `FlowModel.__init__` signature. |
| `flow_model.py`, `model_training.py`, `file_utils.py` TensorFlow code | **Static only** (py_compile, flake8). Never executed. |
| `tests/test_inn_mode.py` | Written, **not run**. |
| Training behavior, loss weights, learning rates | **Not tested at all.** Every default is a reasoned starting point, not a tuned value. |
| Keras `ImageDataGenerator` with `classes=` and `class_mode="categorical"` | Not run (standard Keras usage). |
| Docker / SageMaker / Batch runs | Not run. |


## 4. Decisions made along the way that you should review

1. **Loss used for scripts 3, 6, 7.** These rely mainly on `lambda_nll`: the
   exact negative log-likelihood of a flow whose latent y-components are
   Gaussian about the true y (std `y_sigma`) and whose z is N(0, I). It is the
   stand-in for Zhang & Curtis's likelihood term (their eqn 13). Their term as
   printed needs the prior density evaluated inside the loss, which is awkward
   for the truncated and hierarchical priors here; this form needs only (x, y)
   samples. Scripts 6-7 also keep both MMD terms on at weight 1.
2. **Script 3 loss scale.** `lambda_nll` is 49152 (= 128*128*3) so the loss is
   the total NLL per image, the same scale as script 2, letting script 2's
   learning rate and `grad_norm_thresh` carry over. If you change
   `image_shape`, change `lambda_nll` to match.
3. **Script 3 class signal.** The label is 2 of 49,152 dimensions, so the
   likelihood alone gives it little weight. `lambda_y` 1000 and `y_sigma` 0.05
   add pull toward the class vectors. These two are the first things to tune if
   accuracy is poor. Flow-based generative classifiers are known to classify
   worse than ordinary classifiers, so modest accuracy would not by itself mean
   a bug.
4. **Gravity prior damping (differs from the Octave operator).** With the
   operator as written in `gravinvParker.m`, L = [D2; 10 * I], the 10 * I block
   dominates D2 at every wavelength, so the Gaussian prior it implies is nearly
   white noise, not smooth. `prior_damping` exposes that weight and defaults to
   0.05, which gives smooth profiles (correlation half-width about 7 nodes,
   480 m). Set `prior_damping` 10 for the literal Octave operator. With the
   defaults (`prior_scale` 0.02, `prior_mean` 600 m) the prior depth std is
   about 270 m mid-valley and 440 m at the edges.
5. **Truncation by reflection.** Gaussian-prior depths are folded back inside
   `prior_bounds`; rejection sampling is infeasible in 51 dimensions. The
   network's posterior samples are not constrained, so
   `<which>_frac_negative_depth` reports how often they go negative.
6. **The "uniform" prior does not cover the observed data.** Independent
   uniform depths average out, so that prior predicts nearly the same data every
   time (std about 4 mGal around -45 mGal). The synthetic and Parker data both
   fall outside it (99.8th and 100th percentile of prior-predictive Mahalanobis
   distance; about 70 and 93 for the smooth priors). The network would be
   extrapolating, so expect the uniform run to demonstrate that failure, not
   give a usable posterior. Every run logs `<which>_prior_predictive_pctile` and
   prints a warning above 99.
7. **Hierarchical prior.** log10(lambda) is appended to the model vector and
   estimated. "At MAP lambda" results use the `lambda_map_fraction` (default
   20%) of posterior samples nearest the histogram mode of lambda.
8. **Script 4 and 5 loss weights** (`lambda_y` 10, `lambda_z` 300, `lambda_x`
   400) are patterned on the relative weights in the Ardizzone group's public
   toy demos, recalled from memory, not checked against that code. Kernel is
   the inverse multiquadratic of the paper, summed over several widths instead
   of the paper's single width.
9. **Inverse-kinematics convention.** y = (vertical, horizontal), following
   eqns 13-14 of the paper's appendix as printed. `y_targets` uses that order.
10. **GMM labels.** Components at 45, 90, 135, 180 degrees are red, 225 and 270
    green, 315 blue, 360 purple (`file_utils.GMM_LABEL_MAP`). Change the map
    there, or pass `dataset_params={"label_map": [...]}`.


## 5. Known risks in the untested code

1. `FlowModel._inn_losses` calls
   `flow_bijector.inverse_log_det_jacobian(v, event_ndims=1)` for realnvp-based
   (script 1 already does this) and `event_ndims=3` for Glow. The Glow call is
   the less certain one; `TestInnModeGlow` will show it.
2. For Glow, y is the FIRST `y_dim` components of the flattened latent vector.
   Which Glow scale those belong to was not checked. Any components work in
   principle, but coarse-scale ones should classify better. If script 3
   classifies poorly despite tuning, try taking y from the other end
   (`split_latent`, `sample_posterior`, `_inn_losses` all slice from the
   front).
3. `realnvp_permutation="random"` draws its permutations with `np.random` at
   construction and does not save them, so a model rebuilt with `do_train`
   false would not match its weights. The new scripts use `"alternating"`.
   (Pre-existing behavior, not changed.)
4. The realnvp-based bijector needs an even `prod(image_shape)`, hence the
   automatic one-component padding in scripts 6-7 when the input width is odd.
5. `gravprob` is a faithful port including its quadrature: where depth is near
   zero close to a station the trapezoid rule on the log kernel is inaccurate
   (up to about 3 mGal for an all-zero profile). It affects only prior samples
   touching zero near stations.
6. `sm-submit` still passes `IMAGES_PATH` as the SageMaker training input
   channel for every script, as it already did for script 1. Scripts 4-7 never
   read it, but the variable probably still has to hold a valid S3 prefix for
   the job to be accepted.
7. `S3ImageDataGenerator` lists keys with one `list_objects_v2` call, which
   returns at most 1000 keys (pre-existing). It only matters when IMAGES_PATH is
   an `s3://` URI read directly, not for SageMaker's mounted channel.
8. Script 3 holds `num_eval_images` images and their latents in memory at once
   (about 200 MB each per 1000 images at 128x128).


## 6. Design reference

All of scripts 3-7 use the form x <-> [y, z] (frames 3-7 of
`doc/INNfig_3sec.gif`):

- data side:   v = [x (`x_dim`) | input padding], width D = prod(`image_shape`)
- latent side: u = [y (`y_dim`) | z (`z_dim`) | output padding]
- data -> latent is `flow_bijector.inverse` (= `FlowModel.call`); latent -> data
  is `flow_bijector.forward` (= `FlowModel.inverse`), as in scripts 1-2.
- loss = `lambda_y` L_y + `lambda_z` L_z + `lambda_x` L_x + `lambda_pad` L_pad
  + `lambda_nll` NLL / D  (see `FlowModel._inn_losses` and `doc/config.md`).

| # | Dimensions | Loss terms on by default |
|---|---|---|
| 3 | Glow, D = 49152; y = 2, z = 49150 | nll (weight D), y |
| 4 | x = 2, y = 4, z = 2, padded to D = 16 | y, z-MMD, x-MMD, pad |
| 5 | x = 4, y = 2, z = 2, D = 4 | y, z-MMD, x-MMD |
| 6 | x = 51 (52 if hierarchical), D = 52; y = 12, z = 40 | nll, y, z-MMD, x-MMD, pad |
| 7 | x = 63 (64 if hierarchical), D = 64; y = 12, z = 52 | nll, y, z-MMD, x-MMD, pad |

Scripts 5-7 train on standardized x and y; the problem objects in
`forward_models.py` hold the fixed means and stds and convert both ways.
