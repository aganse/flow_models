# Configuration Reference

This guide lists the user-facing parameters exposed by the training scripts:

- **Application 1** &rarr; `train_flowmodels1.py`
- **Application 2** &rarr; `train_flowmodels2.py`

The `App#` column below indicates which script consumes each parameter
(`1`, `2`, or `1,2`). Default values remain defined inside the scripts so this
document focuses on intent and usage.

## Run Parameters

| Parameter | Description | App# |
| --- | --- | --- |
| `output_dir` | Folder for plots, diagnostic exports, and generated samples. | 1,2 |
| `model_dir` | Folder used to save/load model weights and summaries. | 1,2 |
| `dataset` | Name of the dataset to draw training points from (e.g., `moons`, `gmm`, `mvn`, `cats`). | 1,2 |
| `num_gen_sims` | Number of synthetic samples to draw from the trained flow for downstream analysis. | 1,2 |
| `do_train` | When `True`, train from scratch; when `False`, load weights from `model_dir` for inference. | 1,2 |
| `num_outliers_to_highlight` | Count of lowest-density latent points to highlight in both data/latent scatter plots; `0` disables highlighting. | 1 |
| `img2lat_chunk_size` | Mini-batch size for mapping images to latent space to avoid OOM during analysis. | 2 |
| `images_path` | Root directory (or S3 URI) containing image data with `train/`, `val/` splits for image-based runs. | 2 |
| `do_imgs_and_points` | Enables generation of latent scatter plots and regenerated samples after training. | 2 |
| `do_interp` | Toggles latent interpolation experiments between selected training images. | 2 |
| `sampling_mode` | Chooses latent sampling strategy: `"pca"` for PCA-reduced Gaussian draws or `"direct"` for sampling from the base distribution. | 2 |
| `cov_scale` | Scalar multiplier applied to the PCA covariance when `sampling_mode="pca"` to widen/narrow samples. | 2 |
| `pca_n_components` | Number of principal components to keep when computing latent statistics (`None` disables PCA reduction). | 2 |
| `pca_solver` | PCA solver (`"auto"` or `"randomized"`) used when `pca_n_components` is specified. | 2 |
| `regen_source` | Controls which latent points are inverted into images (`"pca_stats"`, `"flow_base"`, or `"train_pts"`). | 2 |

## Training Parameters

| Parameter | Description | App# |
| --- | --- | --- |
| `num_epochs` | Maximum number of training epochs. | 1,2 |
| `batch_size` | Training mini-batch size drawn from the data generator. | 1,2 |
| `reg_level` | L2 regularization strength applied inside the RealNVP coupling networks. | 1,2 |
| `learning_rate` | Optimizer learning rate; can be a scalar or a list configuring an exponential decay schedule. | 1,2 |
| `early_stopping_patience` | Patience (in epochs) for early stopping on training loss; `<=0` disables early stopping. | 1,2 |
| `num_data_input` | Approximate number of training samples consumed per epoch (used to derive `steps_per_epoch`). | 1,2 |
| `augmentation_factor` | Multiplier for virtual dataset size when augmentation is enabled by the data generator. | 1,2 |
| `grad_norm_thresh` | Global gradient-norm clipping threshold; `None` leaves gradients unclipped. | 1,2 |
| `log_scale_clip` | Clips the log-scale output of RealNVP coupling networks to `[-value, value]`; `<=0` or absent disables clipping. | 1 |
| `jit_compile` | Enables/disables XLA JIT compilation for the Keras training step. | 1,2 |
| `tracking_tool` | Metric logging backend (`"mlflow"` or `"tensorboard"`); `None` skips tracking callbacks. | 1,2 |
| `tracking_port` | Port used when launching the selected tracking tool locally. | 1,2 |
| `tracking_expt_name` | Experiment name under which runs are grouped in the tracking tool. | 1,2 |

## Model Architecture Parameters

| Parameter | Description | App# |
| --- | --- | --- |
| `image_shape` | Shape of the input samples; `(2,)` for 2D points or `(H, W, C)` for image inputs. | 1,2 |
| `bijector` | Choice of flow architecture (`"realnvp-based"` or `"glow"`). | 1,2 |
| `validate_args` | Propagated to TensorFlow Probability bijectors to enable argument validation checks. | 1,2 |
| `realnvp_flow_steps` | Number of RealNVP coupling/permute blocks chained in the flow. Only used when `bijector="realnvp-based"`. | 1,2 |
| `realnvp_hidden_layers` | Sizes of hidden Dense layers inside the shift/log-scale networks for each RealNVP coupling block. Only used when `bijector="realnvp-based"`. | 1,2 |
| `realnvp_permutation` | Permutation strategy between RealNVP coupling blocks: `"alternating"` (reverse/identity alternation) or `"random"` (random shuffle per block). Only used when `bijector="realnvp-based"`. | 1,2 |
| `glow_num_blocks` | Number of multi-scale levels in the Glow architecture. For 128×128 images, at least 3 is recommended. Only used when `bijector="glow"`. | 2 |
| `glow_steps_per_block` | Number of flow steps per Glow level. The paper uses 32; 8 is a lighter starting point. Only used when `bijector="glow"`. | 2 |
| `glow_num_hidden` | Number of filters in the Glow coupling CNN at each level. The paper uses 400; 128–256 is a lighter starting point. Only used when `bijector="glow"`. | 2 |

Refer back to the individual training scripts for concrete default settings and
dataset-specific notes.
