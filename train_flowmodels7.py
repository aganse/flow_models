"""Use case 7: ill-posed inverse problem with noise in the data - the same
glacier-depth gravimetry problem as train_flowmodels6.py, now with Gaussian
noise on the gravity measurements, after Zhang & Curtis (2021) Sec 2.2
(their eqns 10-13).

The noise vector becomes part of the network input, x = [model, noise], with
data = forward(model) + noise, so the INN maps [model, noise] <-> [data, z]
(frame 7 of the README diagram).  Sampling z for the observed data then yields
posterior samples of the model (and of the noise realization), with the
posterior over depth profiles appropriately wider than in the noise-free
case.

Everything after the parameter settings is shared with train_flowmodels6.py
(run_glacier_inversion); the differences are the noise settings below.
"""

import warnings

from model_training import load_param_overrides
from train_flowmodels6 import run_glacier_inversion

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def main():

    # Default user parameter settings:
    # Can change here for local non-Docker usage; or change in params/params7.json for Docker use
    # -----------------------
    run_params = {
        "output_dir": "output",  # local artifacts storage area before possibly logging to mlflow
        "model_dir": "model/flowmodels7",  # local model storage area before possibly logging to mlflow
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
        "noise_sigma": 0.5,  # std of Gaussian data noise (mGal)
        "noise_as_input": True,  # noise is a network input: x = [model, noise]
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
        "tracking_expt_name": "flowmodels7",
        "save_model_weights": False,
    }
    model_arch_params = {
        "image_shape": None,  # set automatically below from the problem's dimensions
        "bijector": "realnvp-based",
        # realnvp-based params:
        "realnvp_flow_steps": 12,  # number of realnvp-based affine coupling layers
        "realnvp_hidden_layers": [512, 512],  # nodes/denselayer in affine coupling layers
        # INN (x <-> [y, z]) params:
        "x_dim": None,  # set automatically: 51 depths (+1 if hierarchical prior) + 12 noise values
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
        "mmd_bandwidths": [3.0, 10.0, 30.0],  # kernel widths, in standardized units (~64 dims)
    }
    load_param_overrides(run_params, training_params, model_arch_params)
    run_glacier_inversion(run_params, training_params, model_arch_params)


if __name__ == "__main__":
    main()
