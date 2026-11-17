"""Use case 4: generative classification (b) - the 2D Gaussian-mixture example
of Ardizzone et al 2019 (their Fig 2 and appendix Sec 2).

Eight Gaussian clusters on a circle carry four color labels.  The flow is
trained as an INN, x <-> [y, z]: in the forward direction 2D points x map to
a label vector y (one-hot color) and a 2D standard-Gaussian latent z; in the
reverse direction a chosen label y* plus samples of z generate points of that
color, i.e. samples of p(x | y*).

Structure follows train_flowmodels1.py (same gmm dataset, now with labels).
"""

import os
import warnings

import numpy as np

from file_utils import GMM_LABEL_COLORS, GMM_LABEL_MAP, get_data_generator
from model_training import default_training_sequence, load_param_overrides
import inn_utils
import utils

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def main():

    # Default user parameter settings:
    # Can change here for local non-Docker usage; or change in params/params4.json for Docker use
    # -----------------------
    run_params = {
        "output_dir": "output",  # local artifacts storage area before possibly logging to mlflow
        "model_dir": "model/flowmodels4",  # local model storage area before possibly logging to mlflow
        "dataset": "gmm",  # the labeled 8-component GMM (only option for this script)
        "num_gen_sims": 1000,  # number of points to generate per color label
        "do_train": True,  # true = training, false = inference w existing model in model_dir
        "num_eval_pts": 4000,  # number of labeled points mapped forward for metrics/plots
        "latent_grid_extent": 3.0,  # half-width of z grid for the latent-layout plot
        "latent_grid_size": 120,  # points per side of that z grid
    }
    training_params = {
        "num_epochs": 50,
        "batch_size": 256,  # MMD compares batches, so don't go too small
        "reg_level": 1e-5,  # regularization level for the L2 reg in realNVP hidden layers
        "learning_rate": [1e-3, 450, 0.90],  # [initial_rate, decay_steps, decay_rate]
        #     ~1e-3 -> ~1e-5 over 50 epochs at 390 steps/epoch (as in Ardizzone appendix 8.1)
        # "learning_rate": 1e-4,  # scaler -> constant learning rate; vector of 3 -> lr schedule
        "early_stopping_patience": 15,  # value <=0 turns off early_stopping
        "num_data_input": 100000,  # num training data pts per epoch
        "augmentation_factor": 1,  # set >1 to have augmentation turned on
        "grad_norm_thresh": 25,  # if not None, clip norm of gradients at this thresh
        "log_scale_clip": 2,  # clip log-scale outputs to [-value, value]; <=0 disables
        "jit_compile": True,  # boolean, normally True but sometimes useful in debugging
        "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
        "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
        "tracking_expt_name": "flowmodels4",
        "save_model_weights": False,
    }
    model_arch_params = {
        "image_shape": (16,),  # nominal (zero-padded) width of both sides of the INN
        "bijector": "realnvp-based",
        # realnvp-based params:
        "realnvp_flow_steps": 8,  # number of realnvp-based affine coupling layers
        "realnvp_hidden_layers": [256, 256],  # nodes/denselayer in affine coupling layers
        # INN (x <-> [y, z]) params:
        "x_dim": 2,  # 2D points; remaining 14 input components are padding
        "y_dim": 4,  # one-hot color label
        "z_dim": 2,  # Gaussian latent; remaining 10 output components are padding
        "lambda_y": 10.0,  # weight of supervised label loss
        "lambda_z": 300.0,  # weight of MMD on (y, z) vs p(y)p(z)
        "lambda_x": 400.0,  # weight of MMD on generated x vs true x
        "lambda_pad": 10.0,  # weight of padding losses
        "lambda_nll": 0.0,  # likelihood term off: this script is the pure Ardizzone setup
        "pad_noise_scale": 0.05,  # amplitude of noise written into padding components
        "mmd_bandwidths": [0.2, 1.0, 3.0],  # kernel widths; clusters have std 0.1, z has std 1
    }
    load_param_overrides(run_params, training_params, model_arch_params)
    os.makedirs(run_params["output_dir"], exist_ok=True)
    os.makedirs(run_params["model_dir"], exist_ok=True)
    # List the param settings:
    print("")
    utils.print_run_params(**run_params, **training_params, **model_arch_params)
    flat_dim = int(np.prod(model_arch_params["image_shape"]))
    num_labels = model_arch_params["y_dim"]
    label_colors = list(GMM_LABEL_COLORS[:num_labels])
    outdir = run_params["output_dir"]

    # Get the data
    # ------------
    # Same generator as train_flowmodels1.py, but class_mode="categorical"
    # makes it yield (points, one-hot color labels), and pad_to zero-pads the
    # 2D points out to the INN's nominal width.
    train_generator = get_data_generator(
        dataset=run_params["dataset"],
        batch_size=training_params["batch_size"],
        class_mode="categorical",
        pad_to=flat_dim,
    )
    sample_x, sample_y = next(train_generator)
    print(
        "train_generator test: shapes of one batch (x, y): ",
        sample_x.shape, sample_y.shape, "\n",
    )
    # Quick sanity-check plot of some of the data, colored by label
    eval_x, eval_y = inn_utils.collect_pairs(
        train_generator, run_params["num_eval_pts"]
    )
    eval_pts = eval_x[:, :2]
    eval_labels = np.argmax(eval_y, axis=1)
    datain_plot_path = outdir + "/trainpts_dataspace.png"
    inn_utils.plot_labeled_pts_2d(
        eval_pts,
        eval_labels,
        label_colors,
        plotfile=datain_plot_path,
        side="data",
        title="original train pts in data space, colored by label",
    )
    print("trainpts_dataspace.png written.")

    # Train the model
    # ---------------
    flow_model, history = default_training_sequence(
        train_generator, run_params, training_params, model_arch_params
    )

    # Analyze/plot various model results
    # ----------------------------------
    print("Analyzing/plotting various model results:")
    print("-----------------------------------------")
    metrics = {}
    plot_paths = [datain_plot_path]

    # Forward direction: map labeled points to label space and latent space.
    y_pred, z_pred, pad_pred = flow_model.forward_yz(eval_x)
    pred_labels = np.argmax(y_pred, axis=1)
    metrics["label_accuracy"] = float(np.mean(pred_labels == eval_labels))
    metrics["label_mse"] = float(np.mean((y_pred - eval_y) ** 2))
    metrics.update(inn_utils.latent_stats(z_pred))
    if pad_pred.shape[1] > 0:
        metrics["output_pad_max_abs"] = float(np.abs(pad_pred).max())
    print(f"label accuracy on {eval_x.shape[0]} pts: {metrics['label_accuracy']:.4f}")

    labelspace_plot_path = outdir + "/trainpts_labelspace.png"
    inn_utils.plot_label_space(
        y_pred, eval_labels, label_colors, plotfile=labelspace_plot_path
    )
    plot_paths.append(labelspace_plot_path)
    print("trainpts_labelspace.png written.")

    confusion_plot_path = outdir + "/label_confusion.png"
    inn_utils.plot_confusion_matrix(
        inn_utils.confusion_matrix(eval_labels, pred_labels, num_labels),
        label_colors,
        plotfile=confusion_plot_path,
        title="Predicted vs true color label",
    )
    plot_paths.append(confusion_plot_path)

    # latent space plot: all four colors should land on the same N(0, I)
    latent_plot_path = outdir + "/trainpts_latentspace.png"
    inn_utils.plot_labeled_pts_2d(
        z_pred,
        eval_labels,
        label_colors,
        plotfile=latent_plot_path,
        side="latent",
        title="mapped train pts in latent (z) space, colored by label",
        unit_circles=True,
    )
    plot_paths.append(latent_plot_path)
    print("trainpts_latentspace.png written.")

    # Invertibility check: forward then backward should reproduce the inputs.
    roundtrip = flow_model.inverse(flow_model.call(eval_x[:500])).numpy()
    metrics["invertibility_max_abs_err"] = float(
        np.abs(roundtrip - eval_x[:500]).max()
    )

    # Reverse direction: for each color label y*, sample z and generate x,
    # i.e. sample the posterior p(x | y*)  (Ardizzone Fig 2).
    sim_pts = []
    sim_labels = []
    for k in range(num_labels):
        y_star = np.zeros(num_labels, dtype=np.float32)
        y_star[k] = 1.0
        pts_k = flow_model.sample_posterior(y_star, run_params["num_gen_sims"])
        sim_pts.append(pts_k)
        sim_labels.append(np.full(pts_k.shape[0], k))
        true_k = eval_pts[eval_labels == k]
        if true_k.shape[0] > 0:
            metrics[f"sim_vs_true_mmd_{label_colors[k]}"] = inn_utils.mmd_numpy(
                pts_k, true_k, bandwidths=model_arch_params["mmd_bandwidths"]
            )
    sim_pts = np.concatenate(sim_pts, axis=0)
    sim_labels = np.concatenate(sim_labels, axis=0)
    dataout_plot_path = outdir + "/simpts_dataspace.png"
    inn_utils.plot_labeled_pts_2d(
        sim_pts,
        sim_labels,
        label_colors,
        plotfile=dataout_plot_path,
        side="data",
        title="generated pts x ~ p(x | label), colored by requested label",
    )
    plot_paths.append(dataout_plot_path)
    print("simpts_dataspace.png written.")

    # Latent-space layout for each fixed label (Ardizzone Fig 8): which mode
    # does each latent position z map to?
    n_side = int(run_params["latent_grid_size"])
    extent = float(run_params["latent_grid_extent"])
    axis_vals = np.linspace(-extent, extent, n_side)
    z1, z2 = np.meshgrid(axis_vals, axis_vals)
    z_grid = np.stack([z1.ravel(), z2.ravel()], axis=1)
    x_grids = []
    for k in range(num_labels):
        y_star = np.zeros((z_grid.shape[0], num_labels), dtype=np.float32)
        y_star[:, k] = 1.0
        x_k = flow_model.sample_posterior(y_star, z=z_grid, y_noise_scale=0.0)
        x_grids.append(x_k.reshape(n_side, n_side, 2))
    layout_plot_path = outdir + "/latent_layout.png"
    inn_utils.plot_latent_layout(
        x_grids,
        _gmm_mode_centers(),
        extent,
        plotfile=layout_plot_path,
        titles=[f"y* = {c}" for c in label_colors],
    )
    plot_paths.append(layout_plot_path)
    print("latent_layout.png written.")

    print("Post-training metrics:")
    for key, value in metrics.items():
        print(f"    {key:<32} {value: .6f}")

    if training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open"):
        import mlflow
        mlflow.log_metrics(metrics)
        for artifact_path in plot_paths:
            mlflow.log_artifact(artifact_path, artifact_path="plots")
        mlflow.log_artifact(run_params["model_dir"], artifact_path="model")
        mlflow.end_run()
        run_params["mlflow_run_open"] = False


def _gmm_mode_centers():
    """Centers of the 8 mixture components, matching the "gmm" dataset
    definition in file_utils.get_data_generator (radius 14, then scaled /10).
    Component k carries color label GMM_LABEL_MAP[k]."""
    n_means = len(GMM_LABEL_MAP)
    radius = 14
    ths = 2 * np.pi / n_means * (np.arange(1, n_means + 1))
    return np.column_stack([radius * np.cos(ths), radius * np.sin(ths)]) / 10


if __name__ == "__main__":
    main()
