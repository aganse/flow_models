import warnings

import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture

import utils
from flow_model import default_training_sequence


warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these

run_params = {
    "output_dir": "output",
    "model_dir": "models/flowmodels1",
    "do_train": True,  # true = training, false = inference w existing model in model_dir
    "use_tensorboard": True,
    "data": "moons",  # "moons", "GMM", "cats", "catsdogs", "invkin", "glacgrav"
    "num_gen_sims": 1000,  # number of new simulated data to generate
}
training_params = {
    "num_epochs": 10,
    "batch_size": 64,
    "reg_level": 0,  # 0.01  # regularization level for the L2 reg in realNVP hidden layers
    "learning_rate": 0.0001,  # scaler -> constant rate; list-of-3 -> exponential decay
    # "learning_rate": [0.001, 500, 0.95],  # [initial_rate, decay_steps, decay_rate]
    "early_stopping_patience": 0,  # value <=0 turns off early_stopping
    "num_data_input": 10000,  # num training data pts or images (whether pts or files)
    "augmentation_factor": 1,  # set >1 to have augmentation turned on
}
model_arch_params = {
    "image_shape": (2,),  # 2D points with (no color in this run)
    "hidden_layers": [512, 512],  # nodes per layer within affine coupling layers
    "flow_steps": 4,  # number of affine coupling layers
    "validate_args": True,
}
# List the param settings:
utils.print_run_params(**run_params, **training_params, **model_arch_params)


if run_params["data"] == "moons":

    def create_train_generator(batch_size=32, noise=0.1):
        while True:
            X, _ = datasets.make_moons(n_samples=batch_size, noise=noise)
            yield X.astype(np.float32)

    train_generator = create_train_generator(training_params["batch_size"])

elif run_params["data"] == "GMM":

    def create_train_generator(
        batch_size=32,
        means=[[0, 0], [3, 3], [-3, -3]],
        covariances=[np.eye(2) for _ in range(3)],
        weights=[0.3, 0.4, 0.3],
    ):
        gmm = GaussianMixture(n_components=len(means))
        gmm.weights_ = np.array(weights)
        gmm.means_ = np.array(means)
        gmm.covariances_ = np.array(covariances)
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
        while True:
            X, _ = gmm.sample(batch_size)
            yield X.astype(np.float32)

    train_generator = create_train_generator(training_params["batch_size"])

else:
    print("Error: invalid run_params['data'] which should be 'moons' or 'GMM'.")

print("train_generator test: shape of one batch: ", next(train_generator).shape, "\n")
input_data_test = np.concatenate([next(train_generator) for _ in range(20)], axis=0)
utils.plot_pts_2d(
    input_data_test, plotfile=run_params["output_dir"] + "/test_input_dataspace.png"
)
print("test_input_dataspace.png written.")


flow_model = default_training_sequence(
    train_generator, run_params, training_params, model_arch_params
)


# Map 1000 pts from train_generator thru flow_model to latent space:
mapped_training_pts, mean, cov, pca, top_outliers, closest_to_mean = (
    utils.imgs_to_gaussian_pts(flow_model, train_generator, 1000)
)
# Latent space plot:
utils.plot_pts_2d(
    mapped_training_pts,
    plotfile=run_params["output_dir"] + "/test_output_latentspace.png",
)
print("test_output_latentspace.png written.")
# Map num_gen_sims sim pts from latent space thru flow_model to data space:
sim_pts = utils.generate_sim_pts(
    flow_model,
    run_params["num_gen_sims"],
    mean,
    cov,
    pca,
    regen_pts=mapped_training_pts,
)
# Data space plot:
utils.plot_pts_2d(
    sim_pts,
    plotfile=run_params["output_dir"] + "/test_output_dataspace.png",
)
print("test_output_dataspace.png written.")
