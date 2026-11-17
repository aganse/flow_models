"""Use case 6: ill-posed inverse problem ignoring noise - the glacier-depth
gravimetry problem of Parker (1994, ch 5), solved in Bayesian form with an
invertible neural network after Zhang & Curtis (2021).

Twelve gravity-anomaly measurements across a glacier-filled valley are to be
explained by the glacier's depth profile h(x) at 51 nodes.  Many profiles fit
the data.  Instead of a single regularized solution (the Occam's inversion of
the original gravinvParker.m), the flow is trained as an INN on samples from a
prior over depth profiles and their forward-modelled data, x <-> [y, z]:
forward, a depth profile maps to its 12 data values y and a Gaussian latent z;
in reverse, the observed data y* plus samples of z give samples of the
posterior distribution over depth profiles.

In this use case there is no data noise anywhere (the "noise-free" form of
Zhang & Curtis Sec 2.2).  train_flowmodels7.py is the same problem with noise,
and reuses run_glacier_inversion() from this file.
"""

import os
import warnings

import numpy as np

from file_utils import get_data_generator
from forward_models import GRAV_X, GRAV_XSTN, GlacGravProblem
from model_training import default_training_sequence, load_param_overrides
import inn_utils
import utils

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def main():

    # Default user parameter settings:
    # Can change here for local non-Docker usage; or change in params/params6.json for Docker use
    # -----------------------
    run_params = {
        "output_dir": "output",  # local artifacts storage area before possibly logging to mlflow
        "model_dir": "model/flowmodels6",  # local model storage area before possibly logging to mlflow
        "dataset": "glacgrav",  # glacier gravimetry (only option for this script)
        "num_gen_sims": 2000,  # number of posterior samples to generate per observed data set
        "do_train": True,  # true = training, false = inference w existing model in model_dir
        # Prior over depth profiles:
        "prior_type": "smooth_gaussian",  # "smooth_gaussian", "smooth_gaussian_hierarchical", or "uniform"
        "prior_scale": 0.02,  # lambda, the prior strength: scalar for "smooth_gaussian";
        #     [low, high] range for "smooth_gaussian_hierarchical" (a scalar s there means [s/3, 3s])
        "prior_mean": 600.0,  # depth (m) the Gaussian prior is centered on
        "prior_damping": 0.05,  # weight of I relative to the 2nd-difference operator in the
        #     prior precision; 10 reproduces gravinvParker.m's operator (nearly white prior)
        "prior_bounds": [0.0, 10000.0],  # depth bounds (m) for the Gaussian priors
        "uniform_bounds": [0.0, 1500.0],  # depth bounds (m) for the "uniform" prior
        # Data noise:
        "noise_sigma": 0.0,  # std of data noise (mGal); 0 in this noise-free use case
        "noise_as_input": False,  # True makes noise a network input (use case 7)
        # Observed data to invert:
        "observed_data": ["synthetic", "parker"],  # sin^2 synthetic truth and/or Parker's measurements
        "observed_seed": 0,  # seed of the noise realization added to synthetic data (if noise)
        "lambda_map_fraction": 0.2,  # hierarchical prior: fraction of samples nearest the
        #     MAP of lambda used for the "at MAP lambda" depth marginals
        "num_test_cases": 200,  # prior-drawn test cases for calibration/re-simulation metrics
        "num_test_post": 200,  # posterior samples per test case for those metrics
    }
    training_params = {
        "num_epochs": 100,
        "batch_size": 512,
        "reg_level": 1e-5,  # regularization level for the L2 reg in realNVP hidden layers
        "learning_rate": [5e-4, 1000, 0.90],  # [initial_rate, decay_steps, decay_rate]
        # "learning_rate": 1e-4,  # scaler -> constant learning rate; vector of 3 -> lr schedule
        "early_stopping_patience": 20,  # value <=0 turns off early_stopping
        "num_data_input": 100000,  # num training (model, data) pairs per epoch, simulated on the fly
        "augmentation_factor": 1,  # set >1 to have augmentation turned on
        "grad_norm_thresh": 25,  # if not None, clip norm of gradients at this thresh
        "log_scale_clip": 2,  # clip log-scale outputs to [-value, value]; <=0 disables
        "jit_compile": True,  # boolean, normally True but sometimes useful in debugging
        "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
        "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
        "tracking_expt_name": "flowmodels6",
        "save_model_weights": False,
    }
    model_arch_params = {
        "image_shape": None,  # set automatically below from the problem's dimensions
        "bijector": "realnvp-based",
        # realnvp-based params:
        "realnvp_flow_steps": 12,  # number of realnvp-based affine coupling layers
        "realnvp_hidden_layers": [512, 512],  # nodes/denselayer in affine coupling layers
        # INN (x <-> [y, z]) params:
        "x_dim": None,  # set automatically: 51 depths (+1 if hierarchical prior) (+12 if noise input)
        "y_dim": None,  # set automatically: 12 data values
        "z_dim": None,  # set automatically: everything else on the latent side
        "lambda_y": 1.0,  # weight of supervised data-prediction loss
        "lambda_z": 1.0,  # weight of MMD on (y, z) vs p(y)p(z)
        "lambda_x": 1.0,  # weight of MMD on generated x vs prior x
        "lambda_pad": 1.0,  # weight of padding losses (only if input width is odd)
        "lambda_nll": 1.0,  # weight of likelihood term (the main term at this dimensionality)
        "y_sigma": 0.02,  # std of predicted data about true data in the likelihood term,
        #     in units of the data's prior std (~9 mGal), so 0.02 ~ 0.2 mGal
        "pad_noise_scale": 0.05,  # amplitude of noise written into padding components
        "mmd_bandwidths": [3.0, 10.0, 30.0],  # kernel widths, in standardized units (~50-60 dims)
    }
    load_param_overrides(run_params, training_params, model_arch_params)
    run_glacier_inversion(run_params, training_params, model_arch_params)


def run_glacier_inversion(run_params, training_params, model_arch_params):  # noqa: C901
    """Everything after parameter loading; shared with train_flowmodels7.py."""

    os.makedirs(run_params["output_dir"], exist_ok=True)
    os.makedirs(run_params["model_dir"], exist_ok=True)
    outdir = run_params["output_dir"]
    num_sims = int(run_params["num_gen_sims"])

    # Set up the problem
    # ------------------
    # The problem object holds the forward model (port of gravprob.m), the
    # prior over depth profiles, the noise model, and the fixed standardization
    # of inputs and data.
    problem = GlacGravProblem(
        prior_type=run_params["prior_type"],
        prior_scale=run_params["prior_scale"],
        prior_mean=run_params["prior_mean"],
        prior_damping=run_params["prior_damping"],
        prior_bounds=run_params["prior_bounds"],
        uniform_bounds=run_params["uniform_bounds"],
        noise_sigma=run_params["noise_sigma"],
        noise_as_input=run_params["noise_as_input"],
    )
    # INN dimensions follow from the problem.  The realnvp-based bijector needs
    # an even width, so an odd input width gets one padding component.
    flat_dim = problem.x_dim + (problem.x_dim % 2)
    model_arch_params["image_shape"] = (flat_dim,)
    model_arch_params["x_dim"] = problem.x_dim
    model_arch_params["y_dim"] = problem.y_dim
    model_arch_params["z_dim"] = flat_dim - problem.y_dim
    # List the param settings:
    print("")
    utils.print_run_params(**run_params, **training_params, **model_arch_params)
    print(
        f"Problem dimensions: {problem.n_depth} depths"
        + (" + log10(lambda)" if problem.has_lambda else "")
        + (f" + {problem.y_dim} noise values" if problem.has_noise else "")
        + f" = x_dim {problem.x_dim}; INN width {flat_dim}; "
        f"y_dim {problem.y_dim}; z_dim {flat_dim - problem.y_dim}\n"
    )

    # Get the data
    # ------------
    # Training pairs (model [+ noise], data) are simulated on the fly from the
    # prior, standardized, and zero-padded to the INN width.
    train_generator = get_data_generator(
        dataset=run_params["dataset"],
        batch_size=training_params["batch_size"],
        problem=problem,
        pad_to=flat_dim,
    )
    sample_x, sample_y = next(train_generator)
    print(
        "train_generator test: shapes of one batch (x, y): ",
        sample_x.shape, sample_y.shape, "\n",
    )
    plot_paths = []
    # Quick sanity-check plot of the prior over depth profiles
    prior_x = problem.sample_inputs(2000)
    prior_depths = problem.depths(prior_x)
    prior_data = problem.forward(prior_x)
    print(
        "Prior depth std (m): "
        f"mid-valley {prior_depths[:, problem.n_depth // 2].std():.0f}, "
        f"edge {prior_depths[:, 0].std():.0f}; "
        f"prior data std (mGal): {problem.y_std.mean():.1f}\n"
    )
    prior_plot_path = outdir + "/prior_depths.png"
    inn_utils.plot_depth_posterior(
        GRAV_X,
        prior_depths,
        plotfile=prior_plot_path,
        stations=GRAV_XSTN,
        true_depths=problem.true_depths(),
        title=f"Prior over depth profiles ({run_params['prior_type']})",
        band_label="prior",
    )
    plot_paths.append(prior_plot_path)
    print("prior_depths.png written.")

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
    n_model = problem.n_model  # depths (+ log10 lambda); excludes any noise inputs

    # Forward direction on a prior-drawn test set (standardized values): how
    # well does the network reproduce the physics, and is z Gaussian?
    num_test = int(run_params["num_test_cases"])
    test_xs, test_ys = inn_utils.collect_pairs(train_generator, num_test)
    y_pred, z_pred, _ = flow_model.forward_yz(test_xs)
    forward_err = problem.unstandardize_y(y_pred) - problem.unstandardize_y(test_ys)
    metrics["forward_rmse_mgal"] = float(np.sqrt(np.mean(forward_err ** 2)))
    metrics.update(inn_utils.latent_stats(z_pred))
    roundtrip = flow_model.inverse(flow_model.call(test_xs)).numpy()
    metrics["invertibility_max_abs_err"] = float(np.abs(roundtrip - test_xs).max())

    latent_plot_path = outdir + "/testpts_latentspace.png"
    utils.plot_pts_2d(
        z_pred,
        main_pts_label="mapped test pts (z)",
        side="latent",
        plotfile=latent_plot_path,
    )
    plot_paths.append(latent_plot_path)
    print("testpts_latentspace.png written.")

    # Posterior quality over the test set: re-simulation misfit and calibration.
    num_post = int(run_params["num_test_post"])
    post_xs = flow_model.sample_posterior(np.repeat(test_ys, num_post, axis=0))
    resim = problem.forward(problem.unstandardize_x(post_xs))
    resim_err = resim - np.repeat(problem.unstandardize_y(test_ys), num_post, axis=0)
    metrics["resim_rmse_mgal"] = float(np.sqrt(np.mean(resim_err ** 2)))
    q, q_inliers, cal_err = inn_utils.calibration_curve(
        test_xs[:, :n_model],
        post_xs.reshape(num_test, num_post, problem.x_dim)[:, :, :n_model],
    )
    metrics["calibration_err_median"] = float(np.median(cal_err))
    metrics["calibration_err_max"] = float(np.max(cal_err))
    calibration_plot_path = outdir + "/calibration.png"
    inn_utils.plot_calibration(
        q,
        q_inliers,
        plotfile=calibration_plot_path,
        title="Posterior calibration over model parameters",
    )
    plot_paths.append(calibration_plot_path)
    print("calibration.png written.")

    # Reverse direction: posterior over depth profiles for each observed data set.
    for which in run_params["observed_data"]:
        d_obs = problem.observed_data(which, seed=run_params["observed_seed"])
        true_depths = problem.true_depths() if which == "synthetic" else None
        print(f"Sampling posterior for '{which}' data: {np.round(d_obs, 1).tolist()}")
        # Is this data vector inside the range of data the prior can produce
        # (i.e. inside what the network was trained on)?
        pctile = inn_utils.prior_predictive_percentile(prior_data, d_obs)
        metrics[f"{which}_prior_predictive_pctile"] = pctile
        if pctile > 99.0:
            print(
                f"    WARNING: '{which}' data lie outside the data predicted by "
                f"this prior (percentile {pctile:.1f}); the posterior below is "
                "an extrapolation of the network and should not be trusted."
            )
        post_x = problem.unstandardize_x(
            flow_model.sample_posterior(problem.standardize_y(d_obs), num_sims)
        )
        post_depths = problem.depths(post_x)
        # data predicted by the posterior models (without their noise part):
        d_pred = problem.forward_noise_free(post_x)
        misfit = d_pred - d_obs[None, :]
        metrics[f"{which}_misfit_rmse_mgal"] = float(np.sqrt(np.mean(misfit ** 2)))
        metrics[f"{which}_depth_std_mean_m"] = float(post_depths.std(axis=0).mean())
        metrics[f"{which}_frac_negative_depth"] = float(np.mean(post_depths < 0))
        if true_depths is not None:
            lo, hi = np.percentile(post_depths, [5, 95], axis=0)
            metrics[f"{which}_depth_rmse_m"] = float(
                np.sqrt(np.mean((post_depths.mean(axis=0) - true_depths) ** 2))
            )
            metrics[f"{which}_truth_coverage_90"] = float(
                np.mean((true_depths >= lo) & (true_depths <= hi))
            )

        extra_sets = None
        if problem.has_lambda:
            # lambda was estimated along with the depths: show its posterior,
            # and the depth marginals for the samples nearest its MAP.
            post_log_lam = problem.log10_lambda(post_x)
            lam_map, near_map = inn_utils.map_window(
                post_log_lam, frac=run_params["lambda_map_fraction"]
            )
            metrics[f"{which}_log10_lambda_map"] = lam_map
            metrics[f"{which}_log10_lambda_mean"] = float(post_log_lam.mean())
            metrics[f"{which}_log10_lambda_std"] = float(post_log_lam.std())
            lambda_plot_path = outdir + f"/lambda_posterior_{which}.png"
            inn_utils.plot_lambda_posterior(
                problem.log10_lambda(prior_x),
                post_log_lam,
                plotfile=lambda_plot_path,
                map_value=lam_map,
                title=f"Prior strength lambda, '{which}' data",
            )
            plot_paths.append(lambda_plot_path)
            extra_sets = {"posterior mean at MAP lambda": post_depths[near_map]}
            map_plot_path = outdir + f"/depth_posterior_{which}_at_map_lambda.png"
            inn_utils.plot_depth_posterior(
                GRAV_X,
                post_depths[near_map],
                plotfile=map_plot_path,
                stations=GRAV_XSTN,
                true_depths=true_depths,
                prior_depths=prior_depths,
                title=(
                    f"Depth posterior at MAP lambda (log10 = {lam_map:.2f}), "
                    f"'{which}' data"
                ),
            )
            plot_paths.append(map_plot_path)

        depth_plot_path = outdir + f"/depth_posterior_{which}.png"
        inn_utils.plot_depth_posterior(
            GRAV_X,
            post_depths,
            plotfile=depth_plot_path,
            stations=GRAV_XSTN,
            true_depths=true_depths,
            prior_depths=prior_depths,
            extra_sets=extra_sets,
            title=f"Posterior over depth profiles, '{which}' data",
        )
        plot_paths.append(depth_plot_path)
        fit_plot_path = outdir + f"/data_fit_{which}.png"
        inn_utils.plot_data_fit(
            GRAV_XSTN,
            d_obs,
            d_pred,
            plotfile=fit_plot_path,
            noise_sigma=problem.noise_sigma,
            title=f"Data predicted by posterior models, '{which}' data",
        )
        plot_paths.append(fit_plot_path)
        std_plot_path = outdir + f"/depth_std_{which}.png"
        inn_utils.plot_depth_std(
            GRAV_X,
            {"prior": prior_depths, "posterior": post_depths},
            plotfile=std_plot_path,
            title=f"Depth uncertainty, '{which}' data",
        )
        plot_paths.append(std_plot_path)
        print(f"depth_posterior_{which}.png, data_fit_{which}.png, depth_std_{which}.png written.")

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


if __name__ == "__main__":
    main()
