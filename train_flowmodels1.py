import os
import warnings

import numpy as np
import tensorflow as tf

from file_utils import get_data_generator
from flow_model import default_training_sequence
import utils

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def main():

    # Default user parameter settings:
    # Can change here for local non-Docker usage; or change in params/params1.json for Docker use
    # -----------------------
    run_params = {
        "output_dir": "output",  # local artifacts storage area before possibly logging to mlflow
        "model_dir": "model/flowmodels1",  # local model storage area before possibly logging to mlflow
        "dataset": "mvn",  # "moons", "gmm", "mvn"
        "num_gen_sims": 1000,  # number of new simulated data to generate
        "do_train": True,  # true = training, false = inference w existing model in model_dir
        "num_outliers_to_highlight": 10,  # set >0 to highlight lowest-density latent points
    }
    training_params = {
        "num_epochs": 50,
        "batch_size": 128,
        "reg_level": 1e-5,  # 0.01  # regularization level for the L2 reg in realNVP hidden layers
        "learning_rate": 1e-4,  # scaler -> constant learning rate; vector of 3 -> lr schedule
        # "learning_rate": [1e-4, 750, 0.90],  # [initial_rate, decay_steps, decay_rate]
        #     steps_per_epoch = num_data_input//batch_size
        #     step = steps_per_epoch * desired_epoch_of_decayed_lr
        #     decayed_lr = initial_rate * decay_rate ^ (step / decay_steps)
        #     ie decay_steps = step * ln(decay_rate) / ln(decayed_lr / initial_rate)
        "early_stopping_patience": 10,  # value <=0 turns off early_stopping
        "num_data_input": 100000,  # num training data pts or images (whether pts or files)
        "augmentation_factor": 1,  # set >1 to have augmentation turned on
        "grad_norm_thresh": 25,  # if not None, clip norm of gradients at this thresh
        "log_scale_clip": 5,  # clip log-scale outputs to [-value, value]; <=0 disables
        "jit_compile": True,  # boolean, normally True but sometimes useful in debugging
        "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
        "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
        "tracking_expt_name": "flowmodels1",
        "save_model_weights": False,
    }
    model_arch_params = {
        "image_shape": (2,),  # 2D points with (no color labels in this run)
        "bijector": "realnvp-based",
        # realnvp-based params:
        "realnvp_flow_steps": 12,  # 8 number of realnvp-based affine coupling layers
        "realnvp_hidden_layers": [512, 512, 512],  # 256,256 nodes/denselayer or filters/cnnlayer in affine coupling layers
        "validate_args": True,
    }
    utils.load_param_overrides(run_params, training_params, model_arch_params)
    # List the param settings:
    print("")
    utils.print_run_params(**run_params, **training_params, **model_arch_params)
    os.makedirs(run_params["output_dir"], exist_ok=True)
    os.makedirs(run_params["model_dir"], exist_ok=True)
    highlight_count = int(run_params.get("num_outliers_to_highlight", 0) or 0)

    # Get the data
    # ------------
    train_generator = get_data_generator(
        dataset=run_params["dataset"],
        batch_size=training_params["batch_size"],
    )
    sample_batch = utils.unwrap_batch(next(train_generator))
    print("train_generator test: shape of one batch: ", sample_batch.shape, "\n")
    datain_plot_path = None
    if run_params["dataset"] in ["moons", "gmm", "mvn"]:
        datain_plot_path = run_params["output_dir"] + "/trainpts_dataspace.png"
        if highlight_count <= 0:
            # Quick sanity-check plot of some of the data for this group of 2D problems
            input_data_test = np.concatenate(
                [utils.unwrap_batch(next(train_generator)) for _ in range(20)], axis=0
            )
            utils.plot_pts_2d(
                input_data_test,
                main_pts_label="original train pts",
                side="data",
                plotfile=datain_plot_path,
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
    # map 1000 pts from train_generator thru flow_model to latent space:
    mapping_results = utils.imgs_to_gaussian_pts(
        flow_model,
        train_generator,
        1000,
        neigvals=None,  # no pca for this 2D problem
        return_input_samples=highlight_count > 0,
    )
    if highlight_count > 0:
        (
            mapped_training_pts,
            mean,
            cov,
            pca,
            top_outliers,
            closest_to_mean,
            mapped_training_inputs,
        ) = mapping_results
    else:
        (
            mapped_training_pts,
            mean,
            cov,
            pca,
            top_outliers,
            closest_to_mean,
        ) = mapping_results
        mapped_training_inputs = None

    train_highlight_latent_pts = None
    train_highlight_data_pts = None
    train_highlight_label = None
    if highlight_count > 0:
        effective_highlight = min(highlight_count, mapped_training_pts.shape[0])
        base_dist = flow_model.flow.distribution
        latent_log_probs = base_dist.log_prob(
            tf.convert_to_tensor(mapped_training_pts, dtype=tf.float32)
        ).numpy()
        highlight_indices = np.argsort(latent_log_probs)[:effective_highlight]
        train_highlight_latent_pts = mapped_training_pts[highlight_indices]
        train_highlight_data_pts = mapped_training_inputs[highlight_indices]
        train_highlight_label = (
            f"lowest {effective_highlight} latent\n"
            "density pts"
        )

    # latent space plot:
    latent_plot_path = run_params["output_dir"] + "/trainpts_latentspace.png"
    utils.plot_pts_2d(
        mapped_training_pts,
        main_pts_label="mapped train pts",
        side="latent",
        plotfile=latent_plot_path,
        highlight_pts=train_highlight_latent_pts,
        highlight_label=train_highlight_label or "latent outliers",
        highlight_color="orange",
    )
    print("trainpts_latentspace.png written.")
    if (
        highlight_count > 0
        and datain_plot_path
        and train_highlight_data_pts is not None
    ):
        utils.plot_pts_2d(
            mapped_training_inputs,
            main_pts_label="original train pts",
            side="data",
            plotfile=datain_plot_path,
            highlight_pts=train_highlight_data_pts,
            highlight_label=train_highlight_label or "latent outliers",
            highlight_color="orange",
        )
        print("trainpts_dataspace.png written.")
    # map num_gen_sims sim pts from latent space thru flow_model to data space:
    sim_pts, sim_latent_pts = utils.generate_sim_pts(
        flow_model,
        run_params["num_gen_sims"],
        sampling_mode="direct",
    )
    sim_highlight_latent_pts = None
    sim_highlight_data_pts = None
    sim_highlight_label = None
    if highlight_count > 0:
        effective_sim_highlight = min(highlight_count, sim_latent_pts.shape[0])
        sim_latent_log_probs = base_dist.log_prob(
            tf.convert_to_tensor(sim_latent_pts, dtype=tf.float32)
        ).numpy()
        sim_highlight_indices = np.argsort(sim_latent_log_probs)[:effective_sim_highlight]
        sim_highlight_latent_pts = sim_latent_pts[sim_highlight_indices]
        sim_highlight_data_pts = sim_pts[sim_highlight_indices]
        sim_highlight_label = (
            f"lowest {effective_sim_highlight} simulated\n"
            "latent density pts"
        )

    sim_latent_plot_path = run_params["output_dir"] + "/simpts_latentspace.png"
    utils.plot_pts_2d(
        sim_latent_pts,
        main_pts_label="simulated latent pts",
        side="latent",
        plotfile=sim_latent_plot_path,
        highlight_pts=sim_highlight_latent_pts,
        highlight_label=sim_highlight_label or "sim latent outliers",
        highlight_color="lightgreen",
    )
    print("simpts_latentspace.png written.")
    # data space plot:
    dataout_plot_path = run_params["output_dir"] + "/simpts_dataspace.png"
    utils.plot_pts_2d(
        sim_pts,
        main_pts_label="mapped sim pts",
        side="data",
        plotfile=dataout_plot_path,
        highlight_pts=sim_highlight_data_pts,
        highlight_label=sim_highlight_label or "sim data outliers",
        highlight_color="lightgreen",
    )
    print("simpts_dataspace.png written.")
    likelihood_sample_count = min(4096, training_params["batch_size"] * 8)
    report_path = os.path.join(
        run_params["model_dir"], "change_of_variables_report.txt"
    )
    likelihood_metrics = _run_change_of_variables_checks(
        flow_model,
        run_params["dataset"],
        training_params["batch_size"],
        num_data_samples=likelihood_sample_count,
        num_latent_samples=likelihood_sample_count,
        report_path=report_path,
    )
    if training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open"):
        import mlflow
        if likelihood_metrics:
            mlflow.log_metrics(likelihood_metrics)
        plot_paths = [latent_plot_path, sim_latent_plot_path, dataout_plot_path]
        if datain_plot_path:
            plot_paths.append(datain_plot_path)
        for artifact_path in plot_paths:
            mlflow.log_artifact(artifact_path, artifact_path="plots")
        mlflow.log_artifact(run_params["model_dir"], artifact_path="model")
        mlflow.end_run()
        run_params["mlflow_run_open"] = False


def _collect_samples_from_generator(generator, num_samples):
    """Accumulate exactly num_samples items from a generator that yields batches."""
    collected = []
    total = 0
    while total < num_samples:
        batch = utils.unwrap_batch(next(generator))
        take = min(num_samples - total, batch.shape[0])
        collected.append(np.asarray(batch[:take], dtype=np.float32))
        total += take
    return np.concatenate(collected, axis=0)


def _describe_stats(label, values):
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"    {label:<20} mean={arr.mean(): .6f}  std={arr.std(): .6f}  "
        f"min={arr.min(): .6f}  max={arr.max(): .6f}  max|.|={np.abs(arr).max(): .6f}"
    )


def _run_change_of_variables_checks(
    flow_model,
    dataset_label,
    batch_size,
    num_data_samples=2048,
    num_latent_samples=2048,
    report_path=None,
):
    """Verify log p(x) consistency via change-of-variables in both directions.

    When report_path is supplied, write the formatted table to disk instead of
    emitting it to stdout.
    """
    flat_dim = int(np.prod(flow_model.image_shape))
    lines = []
    lines.append(
        "Change-of-variables consistency checks to verify model implementation "
        "(but not training success):"
    )

    # Data -> latent direction
    data_gen = get_data_generator(dataset=dataset_label, batch_size=batch_size)
    data_samples = _collect_samples_from_generator(data_gen, num_data_samples)
    data_flat = data_samples.reshape(data_samples.shape[0], flat_dim)
    x_tensor = tf.convert_to_tensor(data_flat, dtype=tf.float32)
    log_px_direct = flow_model.flow.log_prob(x_tensor).numpy()
    z_tensor = flow_model.flow.bijector.inverse(x_tensor)
    log_pz = flow_model.flow.distribution.log_prob(z_tensor).numpy()
    logdet_inverse = flow_model.flow.bijector.inverse_log_det_jacobian(
        x_tensor, event_ndims=1
    ).numpy()
    log_px_via_change = log_pz + logdet_inverse
    diff_data = log_px_direct - log_px_via_change

    lines.append("  data → latent")
    lines.append(_describe_stats("log_px_direct", log_px_direct))
    lines.append(_describe_stats("log_pz+log|detJ|", log_px_via_change))
    lines.append(_describe_stats("difference", diff_data))

    # Latent -> data direction
    base_dist = flow_model.flow.distribution
    z_samples = base_dist.sample(num_latent_samples)
    log_pz_direct = base_dist.log_prob(z_samples).numpy()
    logdet_forward = flow_model.flow.bijector.forward_log_det_jacobian(
        z_samples, event_ndims=1
    ).numpy()
    x_from_z = flow_model.flow.bijector.forward(z_samples)
    log_px_direct_from_z = flow_model.flow.log_prob(x_from_z).numpy()
    log_px_via_change = log_pz_direct - logdet_forward
    diff_latent = log_px_direct_from_z - log_px_via_change
    log_pz_via_change = log_px_direct_from_z + logdet_forward
    diff_pz = log_pz_direct - log_pz_via_change

    lines.append("  latent → data")
    lines.append(_describe_stats("log_px_direct", log_px_direct_from_z))
    lines.append(_describe_stats("log_pz-log|detJ|", log_px_via_change))
    lines.append(_describe_stats("difference", diff_latent))
    lines.append(_describe_stats("log_pz_direct", log_pz_direct))
    lines.append(_describe_stats("log_pz_via_x", log_pz_via_change))
    lines.append(_describe_stats("pz difference", diff_pz))

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
        print(f"Change-of-variables consistency checks written to {report_path}")
    else:
        print("\n".join(lines))

    return {
        "data_latent_diff_mean": float(np.mean(diff_data)),
        "data_latent_diff_max_abs": float(np.abs(diff_data).max()),
        "latent_data_diff_mean": float(np.mean(diff_latent)),
        "latent_data_diff_max_abs": float(np.abs(diff_latent).max()),
        "latent_pz_diff_max_abs": float(np.abs(diff_pz).max()),
    }


if __name__ == "__main__":
    main()
