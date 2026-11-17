"""Use case 5: ill-conditioned parameter estimation - the inverse kinematics
example of Ardizzone et al 2019 (their Fig 3 and appendix Sec 3).

An articulated arm with three segments is mounted on a vertical rail.  Its
four parameters x (rail height and three joint angles) determine the 2D
position y of its end point, but many different x reach the same y.  The flow
is trained as an INN, x <-> [y, z]: forward it predicts the end point y and a
2D Gaussian latent z; in reverse, a target end point y* plus samples of z give
samples of the posterior p(x | y*), the whole family of arm poses reaching y*.
An approximate Bayesian computation (ABC) rejection sampler provides a
reference posterior to compare against.
"""

import os
import warnings

import numpy as np

from file_utils import get_data_generator
from forward_models import InvKinProblem
from model_training import default_training_sequence, load_param_overrides
import inn_utils
import utils

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these

X_NAMES = ["x1 (rail height)", "x2 (angle 1)", "x3 (angle 2)", "x4 (angle 3)"]


def main():  # noqa: C901

    # Default user parameter settings:
    # Can change here for local non-Docker usage; or change in params/params5.json for Docker use
    # -----------------------
    run_params = {
        "output_dir": "output",  # local artifacts storage area before possibly logging to mlflow
        "model_dir": "model/flowmodels5",  # local model storage area before possibly logging to mlflow
        "dataset": "invkin",  # inverse kinematics (only option for this script)
        "num_gen_sims": 2000,  # number of posterior samples to generate per target
        "do_train": True,  # true = training, false = inference w existing model in model_dir
        "y_targets": [[0.0, 1.5], [0.5, 1.6]],  # target end points y* as (vertical, horizontal)
        "ik_sigmas": [0.25, 0.5, 0.5, 0.5],  # prior stds of x (rail height; 3 angles in radians)
        "ik_lengths": [0.5, 0.5, 1.0],  # arm segment lengths
        "do_abc_reference": True,  # compute rejection-ABC reference posteriors to compare with
        "abc_num_prior": 2000000,  # prior draws per target for ABC (keeps closest num_gen_sims)
        "num_test_cases": 300,  # prior-drawn test cases for calibration/re-simulation metrics
        "num_test_post": 200,  # posterior samples per test case for those metrics
    }
    training_params = {
        "num_epochs": 50,
        "batch_size": 512,  # MMD compares batches, so don't go too small
        "reg_level": 1e-5,  # regularization level for the L2 reg in realNVP hidden layers
        "learning_rate": [1e-3, 450, 0.90],  # [initial_rate, decay_steps, decay_rate]
        # "learning_rate": 1e-4,  # scaler -> constant learning rate; vector of 3 -> lr schedule
        "early_stopping_patience": 15,  # value <=0 turns off early_stopping
        "num_data_input": 200000,  # num training data pts per epoch
        "augmentation_factor": 1,  # set >1 to have augmentation turned on
        "grad_norm_thresh": 25,  # if not None, clip norm of gradients at this thresh
        "log_scale_clip": 2,  # clip log-scale outputs to [-value, value]; <=0 disables
        "jit_compile": True,  # boolean, normally True but sometimes useful in debugging
        "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
        "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
        "tracking_expt_name": "flowmodels5",
        "save_model_weights": False,
    }
    model_arch_params = {
        "image_shape": (4,),  # 4 arm parameters; no padding needed (y_dim + z_dim = 4)
        "bijector": "realnvp-based",
        # realnvp-based params:
        "realnvp_flow_steps": 12,  # number of realnvp-based affine coupling layers
        "realnvp_hidden_layers": [256, 256, 256],  # nodes/denselayer in affine coupling layers
        # INN (x <-> [y, z]) params:
        "x_dim": 4,  # arm parameters
        "y_dim": 2,  # end point position
        "z_dim": 2,  # Gaussian latent
        "lambda_y": 10.0,  # weight of supervised end-point loss
        "lambda_z": 300.0,  # weight of MMD on (y, z) vs p(y)p(z)
        "lambda_x": 400.0,  # weight of MMD on generated x vs prior x
        "lambda_pad": 0.0,  # no padding in this problem
        "lambda_nll": 0.0,  # likelihood term off: this script is the pure Ardizzone setup
        "mmd_bandwidths": [0.3, 1.2, 3.0],  # kernel widths, in standardized units
    }
    load_param_overrides(run_params, training_params, model_arch_params)
    os.makedirs(run_params["output_dir"], exist_ok=True)
    os.makedirs(run_params["model_dir"], exist_ok=True)
    # List the param settings:
    print("")
    utils.print_run_params(**run_params, **training_params, **model_arch_params)
    flat_dim = int(np.prod(model_arch_params["image_shape"]))
    outdir = run_params["output_dir"]
    num_sims = int(run_params["num_gen_sims"])

    # Get the data
    # ------------
    # The problem object holds the forward model, the prior, and the fixed
    # standardization of x and y; the generator draws (x, y) training pairs
    # from it (standardized), an unlimited supply since simulation is cheap.
    problem = InvKinProblem(
        sigmas=run_params["ik_sigmas"], lengths=run_params["ik_lengths"]
    )
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
    # Quick sanity-check plot of the prior: arm poses drawn from p(x)
    prior_x = problem.sample_inputs(1000)
    prior_plot_path = outdir + "/prior_arms.png"
    inn_utils.plot_arm_posteriors(
        {"arm poses drawn from the prior p(x)": problem.arm_points(prior_x)},
        None,
        plotfile=prior_plot_path,
    )
    plot_paths.append(prior_plot_path)
    print("prior_arms.png written.")

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

    # Forward direction on a prior-drawn test set (standardized values):
    num_test = int(run_params["num_test_cases"])
    test_xs, test_ys = inn_utils.collect_pairs(train_generator, num_test)
    y_pred, z_pred, _ = flow_model.forward_yz(test_xs)
    forward_err = problem.unstandardize_y(y_pred) - problem.unstandardize_y(test_ys)
    metrics["forward_rmse"] = float(np.sqrt(np.mean(forward_err ** 2)))
    metrics.update(inn_utils.latent_stats(z_pred))
    roundtrip = flow_model.inverse(flow_model.call(test_xs)).numpy()
    metrics["invertibility_max_abs_err"] = float(np.abs(roundtrip - test_xs).max())

    # latent space plot of the mapped test points (should look like N(0, I)):
    latent_plot_path = outdir + "/testpts_latentspace.png"
    utils.plot_pts_2d(
        z_pred,
        main_pts_label="mapped test pts (z)",
        side="latent",
        plotfile=latent_plot_path,
    )
    plot_paths.append(latent_plot_path)
    print("testpts_latentspace.png written.")

    # Posterior quality over the test set, as in Ardizzone Sec 4.2 / Table 2:
    # calibration error, and re-simulation error averaged over the posterior.
    num_post = int(run_params["num_test_post"])
    post_xs = flow_model.sample_posterior(np.repeat(test_ys, num_post, axis=0))
    resim = problem.forward(problem.unstandardize_x(post_xs))
    resim_err = np.linalg.norm(
        resim - np.repeat(problem.unstandardize_y(test_ys), num_post, axis=0),
        axis=1,
    )
    metrics["resim_err_mean"] = float(resim_err.mean())
    metrics["resim_err_median"] = float(np.median(resim_err))
    q, q_inliers, cal_err = inn_utils.calibration_curve(
        test_xs[:, : problem.x_dim],
        post_xs.reshape(num_test, num_post, problem.x_dim),
    )
    metrics["calibration_err_median"] = float(np.median(cal_err))
    for j in range(problem.x_dim):
        metrics[f"calibration_err_x{j + 1}"] = float(cal_err[j])
    calibration_plot_path = outdir + "/calibration.png"
    inn_utils.plot_calibration(
        q, q_inliers, plotfile=calibration_plot_path, names=X_NAMES
    )
    plot_paths.append(calibration_plot_path)
    print("calibration.png written.")

    # Reverse direction for each chosen target end point y*  (Ardizzone Fig 3):
    for k, y_target in enumerate(run_params["y_targets"]):
        tag = f"target{k + 1}"
        y_target = np.asarray(y_target, dtype=np.float64)
        print(f"Sampling posterior for {tag}: y* = {y_target.tolist()}")
        inn_xs = flow_model.sample_posterior(
            problem.standardize_y(y_target), num_sims
        )
        inn_x = problem.unstandardize_x(inn_xs)
        inn_resim = np.linalg.norm(problem.forward(inn_x) - y_target, axis=1)
        metrics[f"{tag}_resim_err_mean"] = float(inn_resim.mean())

        arm_sets = {"INN posterior p(x | y*)": problem.arm_points(inn_x)}
        marginal_sets = {"INN": inn_x}
        if run_params["do_abc_reference"]:
            abc_x, abc_tol = problem.abc_posterior(
                y_target,
                n_keep=num_sims,
                n_prior=int(run_params["abc_num_prior"]),
            )
            print(f"    ABC reference: kept {abc_x.shape[0]} samples, tolerance {abc_tol:.4f}")
            metrics[f"{tag}_abc_tolerance"] = abc_tol
            metrics[f"{tag}_mmd_inn_vs_abc"] = inn_utils.mmd_numpy(
                inn_xs,
                problem.standardize_x(abc_x),
                bandwidths=model_arch_params["mmd_bandwidths"],
            )
            arm_sets["ABC reference posterior"] = problem.arm_points(abc_x)
            marginal_sets["ABC"] = abc_x

        arms_plot_path = outdir + f"/arm_posterior_{tag}.png"
        inn_utils.plot_arm_posteriors(arm_sets, y_target, plotfile=arms_plot_path)
        plot_paths.append(arms_plot_path)
        marginals_plot_path = outdir + f"/marginals_{tag}.png"
        inn_utils.plot_marginals(
            marginal_sets, plotfile=marginals_plot_path, names=X_NAMES
        )
        plot_paths.append(marginals_plot_path)
        print(f"arm_posterior_{tag}.png and marginals_{tag}.png written.")

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
