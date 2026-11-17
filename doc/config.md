# Configuration Reference

This guide lists the user-facing parameters exposed by the training scripts:

- **Application 1** &rarr; `train_flowmodels1.py` (distribution mapping)
- **Application 2** &rarr; `train_flowmodels2.py` (generative image modeling)
- **Application 3** &rarr; `train_flowmodels3.py` (generative classification, cat/dog images)
- **Application 4** &rarr; `train_flowmodels4.py` (generative classification, 2D Gaussian mixture)
- **Application 5** &rarr; `train_flowmodels5.py` (ill-conditioned parameter estimation, inverse kinematics)
- **Application 6** &rarr; `train_flowmodels6.py` (ill-posed inverse problem ignoring noise, glacier gravimetry)
- **Application 7** &rarr; `train_flowmodels7.py` (ill-posed inverse problem with noise, glacier gravimetry)

The `App#` column below indicates which script consumes each parameter. Default
values remain defined inside the scripts (and mirrored in `params/paramsN.json`)
so this document focuses on intent and usage.

Applications 3-7 train the flow in "INN mode": the latent side of the flow is
split into `[y, z]`, where `y` is trained to match known targets (labels, or the
data of a forward problem) and `z` is a Gaussian latent. That mode is switched
on by `y_dim > 0` in the model architecture parameters; see the INN section
below.

## Run Parameters

| Parameter | Description | App# |
| --- | --- | --- |
| `output_dir` | Folder for plots, diagnostic exports, and generated samples. | 1-7 |
| `model_dir` | Folder used to save/load model weights and summaries. | 1-7 |
| `dataset` | Name of the dataset to draw training points from: `moons`, `gmm`, `mvn` (1); `cats` (2); `catsdogs` (3); `gmm` with labels (4); `invkin` (5); `glacgrav` (6,7). | 1-7 |
| `num_gen_sims` | Number of synthetic samples to draw from the trained flow for downstream analysis. In apps 3 and 4 this is per class/label; in apps 5-7 it is the number of posterior samples per target or observed data set. | 1-7 |
| `do_train` | When `True`, train from scratch; when `False`, load weights from `model_dir` for inference. | 1-7 |
| `num_outliers_to_highlight` | Count of lowest-density latent points to highlight in both data/latent scatter plots; `0` disables highlighting. | 1 |
| `img2lat_chunk_size` | Mini-batch size for mapping images to latent space to avoid OOM during analysis. | 2,3 |
| `images_path` | Root directory (or S3 URI) containing image data with `train/`, `val/` splits for image-based runs. App 3 needs class subdirectories (e.g. `cat/`, `dog/`) inside both. | 2,3 |
| `do_imgs_and_points` | Enables the post-training analysis: latent scatter plots and regenerated samples (2); classification of validation images, label-space plots, and generated images (3). | 2,3 |
| `do_interp` | Toggles latent interpolation experiments between selected training images. | 2 |
| `sampling_mode` | Chooses latent sampling strategy: `"pca"` for PCA-reduced Gaussian draws or `"direct"` for sampling from the base distribution. | 2 |
| `cov_scale` | Scalar multiplier applied to the PCA covariance when `sampling_mode="pca"` to widen/narrow samples. | 2 |
| `pca_n_components` | Number of principal components to keep when computing latent statistics (`None` disables PCA reduction). | 2 |
| `pca_solver` | PCA solver (`"auto"` or `"randomized"`) used when `pca_n_components` is specified. | 2 |
| `regen_source` | Controls which latent points are inverted into images (`"pca_stats"`, `"flow_base"`, or `"train_pts"`). | 2 |
| `classes` | Image subdirectory names (within `train/` and `val/`) used as the classes, in label order. Its length must equal `y_dim`. | 3 |
| `num_eval_images` | Number of validation images (and of training images) mapped forward and classified for metrics and plots. | 3 |
| `sample_temperature` | Standard deviation of `z` when generating images; values below 1 trade variety for image quality. | 3 |
| `do_class_swap` | When `True`, regenerates a few validation images from their own latent `z` with each class label in turn. | 3 |
| `num_swap_images` | Number of validation images used for the class-swap figure. | 3 |
| `num_eval_pts` | Number of labeled points mapped forward for metrics and plots. | 4 |
| `latent_grid_extent` | Half-width of the square grid of `z` values used for the latent-layout plot. | 4 |
| `latent_grid_size` | Number of grid points per side for the latent-layout plot. | 4 |
| `y_targets` | List of target end points `y*` as `[vertical, horizontal]` for which posteriors over arm poses are sampled. | 5 |
| `ik_sigmas` | Prior standard deviations of the four arm parameters (rail height, then three joint angles in radians). | 5 |
| `ik_lengths` | Lengths of the three arm segments. | 5 |
| `do_abc_reference` | When `True`, computes a rejection-ABC reference posterior for each target to compare the INN posterior against. | 5 |
| `abc_num_prior` | Number of prior draws per target for the ABC reference; the `num_gen_sims` draws landing closest to the target are kept. | 5 |
| `num_test_cases` | Number of prior-drawn test cases used for the calibration and re-simulation metrics. | 5,6,7 |
| `num_test_post` | Number of posterior samples drawn per test case for those metrics. | 5,6,7 |
| `prior_type` | Prior over depth profiles: `"smooth_gaussian"` (Gaussian whose precision is `prior_scale^2 * L^T L`, with `L` the regularization operator `[D2; prior_damping * I]`), `"smooth_gaussian_hierarchical"` (same, with `prior_scale` itself random and estimated), or `"uniform"` (independent uniform depth per node). | 6,7 |
| `prior_scale` | Prior strength lambda. A scalar for `"smooth_gaussian"`. For `"smooth_gaussian_hierarchical"` a `[low, high]` range over which log10(lambda) is uniform (a scalar `s` is read as `[s/3, 3s]`). Ignored for `"uniform"`. | 6,7 |
| `prior_mean` | Depth (m) on which the Gaussian priors are centered. | 6,7 |
| `prior_damping` | Weight of the identity block relative to the second-difference block in `L`. Small values (default 0.05) give smooth profiles; `10` reproduces the operator of the original Octave script, which gives a nearly uncorrelated prior. | 6,7 |
| `prior_bounds` | `[min, max]` depth (m) for the Gaussian priors; samples are folded back inside by reflection. | 6,7 |
| `uniform_bounds` | `[min, max]` depth (m) for the `"uniform"` prior. | 6,7 |
| `noise_sigma` | Standard deviation (mGal) of Gaussian noise on the gravity data. `0` in app 6. | 6,7 |
| `noise_as_input` | When `True` (app 7) the noise vector is part of the network input, `x = [model, noise]` with data = forward(model) + noise. `False` in app 6. | 6,7 |
| `observed_data` | Which observed data sets to invert: `"synthetic"` (forward model of the sin^2 test profile, plus noise if `noise_sigma > 0`) and/or `"parker"` (Parker's 12 measured anomalies). | 6,7 |
| `observed_seed` | Random seed of the noise realization added to the synthetic observed data. | 6,7 |
| `lambda_map_fraction` | For the hierarchical prior: fraction of posterior samples nearest the MAP of lambda that are used for the "at MAP lambda" depth marginals. | 6,7 |

## Training Parameters

| Parameter | Description | App# |
| --- | --- | --- |
| `num_epochs` | Maximum number of training epochs. | 1-7 |
| `batch_size` | Training mini-batch size drawn from the data generator. In apps 4-7 the MMD loss terms compare whole batches, so very small batches weaken them. | 1-7 |
| `reg_level` | L2 regularization strength applied inside the RealNVP coupling networks. | 1-7 |
| `learning_rate` | Optimizer learning rate; can be a scalar or a list configuring an exponential decay schedule. | 1-7 |
| `early_stopping_patience` | Patience (in epochs) for early stopping on training loss; `<=0` disables early stopping. | 1-7 |
| `num_data_input` | Approximate number of training samples consumed per epoch (used to derive `steps_per_epoch`). In apps 4-7 training pairs are simulated on the fly, so this only sets the epoch length. | 1-7 |
| `augmentation_factor` | Multiplier for virtual dataset size when augmentation is enabled by the data generator. | 1-7 |
| `grad_norm_thresh` | Global gradient-norm clipping threshold; `None` leaves gradients unclipped. | 1-7 |
| `log_scale_clip` | Clips the log-scale output of RealNVP coupling networks to `[-value, value]`; `<=0` or absent disables clipping. | 1,4,5,6,7 |
| `jit_compile` | Enables/disables XLA JIT compilation for the Keras training step. | 1-7 |
| `tracking_tool` | Metric logging backend (`"mlflow"` or `"tensorboard"`); `None` skips tracking callbacks. | 1-7 |
| `tracking_port` | Port used when launching the selected tracking tool locally. | 1-7 |
| `tracking_expt_name` | Experiment name under which runs are grouped in the tracking tool. | 1-7 |
| `save_model_weights` | When `True`, saves model weights to `model_dir` after training; `False` skips the save (useful for fast experimentation). | 1-7 |
| `checkpoint_every_n_epochs` | Save a checkpoint every N epochs for spot-instance resume; `0` disables. Keeps only the 2 most recent checkpoints during training; all are deleted once final weights are saved. | 1-7 |

## Model Architecture Parameters

| Parameter | Description | App# |
| --- | --- | --- |
| `image_shape` | Shape of the input samples; `(2,)` for 2D points or `(H, W, C)` for image inputs. In INN mode with the RealNVP bijector this is the (even) nominal width of both sides of the network, including any padding. Apps 6 and 7 set it automatically from the problem (leave it `None`/`null`). | 1-7 |
| `bijector` | Choice of flow architecture (`"realnvp-based"` or `"glow"`). | 1-7 |
| `validate_args` | Propagated to TensorFlow Probability bijectors to enable argument validation checks. | 1,2 |
| `realnvp_flow_steps` | Number of RealNVP coupling/permute blocks chained in the flow. Only used when `bijector="realnvp-based"`. | 1-7 |
| `realnvp_hidden_layers` | Sizes of hidden Dense layers inside the shift/log-scale networks for each RealNVP coupling block. Only used when `bijector="realnvp-based"`. | 1-7 |
| `realnvp_permutation` | Permutation strategy between RealNVP coupling blocks: `"alternating"` (reverse/identity alternation) or `"random"` (random shuffle per block). Only used when `bijector="realnvp-based"`. | 1-7 |
| `glow_num_blocks` | Number of multi-scale levels in the Glow architecture. For 128×128 images, at least 3 is recommended. Only used when `bijector="glow"`. | 2,3 |
| `glow_steps_per_block` | Number of flow steps per Glow level. The paper uses 32; 8 is a lighter starting point. Only used when `bijector="glow"`. | 2,3 |
| `glow_num_hidden` | Number of filters in the Glow coupling CNN at each level. The paper uses 400; 128–256 is a lighter starting point. Only used when `bijector="glow"`. | 2,3 |

### INN-mode parameters (also in `model_arch_params`)

With `y_dim` absent or `0` the flow is trained as a plain density model (apps 1
and 2) and the remaining parameters in this table are ignored.

Layout: the data side of the flow holds `[x | input padding]` and the latent
side holds `[y | z | output padding]`, each of total width `prod(image_shape)`.

The training loss is

    lambda_y * L_y + lambda_z * L_z + lambda_x * L_x + lambda_pad * L_pad + lambda_nll * NLL / D

where `D = prod(image_shape)`. `L_y`, `L_z`, `L_x` and the padding losses are
those of Ardizzone et al 2019 (Sec 3.3); the likelihood term serves the purpose
of the likelihood term of Zhang & Curtis 2021 (eqn 13) and is the one that
remains effective in high dimensions. Each term is logged per epoch
(`loss_y`, `loss_z`, `loss_x`, `loss_pad`, `nll_per_dim`) alongside `loss`,
`bits_per_dim`, the gradient norms, and `learning_rate`.

| Parameter | Description | App# |
| --- | --- | --- |
| `y_dim` | Number of leading latent-side components that represent `y` (labels or data). `>0` switches on INN mode, in which the data generator must yield `(x, y)` pairs. Set automatically in apps 6 and 7. | 3-7 |
| `z_dim` | Number of Gaussian latent components following `y`. Defaults to all remaining components; anything left over is output padding. Set automatically in apps 6 and 7. | 4-7 |
| `x_dim` | Number of leading data-side components that hold the real input `x`. Defaults to all of them; anything left over is input padding. Not used with Glow. Set automatically in apps 6 and 7. | 4-7 |
| `lambda_y` | Weight of the supervised loss `L_y`, the mean squared error between predicted and true `y`. | 3-7 |
| `lambda_z` | Weight of `L_z`, the MMD between the network's `(y, z)` outputs and `(true y, Gaussian z)`. Enforces that `z` is Gaussian and independent of `y`. | 3-7 |
| `lambda_x` | Weight of `L_x`, the MMD between `x` generated backward from `(true y, sampled z)` and true `x`. Costs one extra backward pass per training step. | 3-7 |
| `lambda_pad` | Weight of the padding losses, which keep padding components near zero and stop information being stored in them. Only matters when padding exists. | 3-7 |
| `lambda_nll` | Weight of the likelihood term: the exact negative log-likelihood per dimension of a flow whose latent `y` components are Gaussian about the true `y` (std `y_sigma`) and whose `z` is standard Gaussian. `0` gives the pure Ardizzone setup (apps 4, 5). Setting it to `prod(image_shape)` makes the term the total NLL per sample, as in apps 1-2 (done in app 3). | 3-7 |
| `y_sigma` | Standard deviation of the `y` components about their targets in the likelihood term, in the (standardized) units of `y`. Also the jitter added to `y*` when sampling posteriors from likelihood-trained models. | 3,6,7 |
| `pad_noise_scale` | Amplitude of the low-level noise written into padding components (padding is "nearly zero", not exactly zero). | 4,6,7 |
| `mmd_bandwidths` | List of widths `h` of the inverse multiquadratic kernels `h^2 / (h^2 + r^2)` summed in the MMD terms, in the units the network sees (standardized in apps 5-7). Should bracket the typical distances between samples. | 4-7 |

Refer back to the individual training scripts for concrete default settings and
dataset-specific notes.
