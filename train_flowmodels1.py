import warnings

import numpy as np

from file_utils import get_data_generator
from flow_model import default_training_sequence
import utils


warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these

run_params = {
    "output_dir": "output",
    "model_dir": "models/flowmodels1",
    "dataset": "gmm",
    "num_gen_sims": 1000,  # number of new simulated data to generate
    "do_train": True,  # true = training, false = inference w existing model in model_dir
    "use_tensorboard": False,
}
training_params = {
    "num_epochs": 20,
    "batch_size": 128,
    "reg_level": 0.0,  # 0.01  # regularization level for the L2 reg in realNVP hidden layers
    "learning_rate": 0.00002,  # scaler -> constant learning rate
    "early_stopping_patience": 0,  # value <=0 turns off early_stopping
    "num_data_input": 1000,  # num training data pts or images (whether pts or files)
    "augmentation_factor": 1,  # set >1 to have augmentation turned on
}
model_arch_params = {
    "image_shape": (2,),  # 2D points with (no color labels in this run)
    "hidden_layers": [256, 256],  # nodes per layer within affine coupling layers
    "flow_steps": 4,  # number of affine coupling layers
    "validate_args": True,
}
# List the param settings:
utils.print_run_params(**run_params, **training_params, **model_arch_params)


# Get the data
# ------------
train_generator = get_data_generator(
    dataset=run_params["dataset"],
    batch_size=training_params["batch_size"],
)
# print("train_generator test: shape of one batch: ", next(train_generator)[0].shape, "\n")
# # just a quick sanity-check plot of some of the data:
# input_data_test = np.concatenate([next(train_generator)[0] for _ in range(20)], axis=0)
# utils.plot_pts_2d(
#     input_data_test,
#     main_pts_label="original train pts",
#     side="data",
#     plotfile=run_params["output_dir"] + "/test_input_dataspace.png"
# )
# print("test_input_dataspace.png written.")


# Train the model
# ---------------
flow_model, history = default_training_sequence(
    train_generator, run_params, training_params, model_arch_params
)


# Analyze/plot various model results
# ----------------------------------
# map 1000 pts from train_generator thru flow_model to latent space:
mapped_training_pts, mean, cov, pca, top_outliers, closest_to_mean = (
    utils.imgs_to_gaussian_pts(flow_model, train_generator, 1000)
)
# latent space plot:
utils.plot_pts_2d(
    mapped_training_pts,
    main_pts_label="mapped train pts",
    side="latent",
    plotfile=run_params["output_dir"] + "/test_output_latentspace.png",
)
print("test_output_latentspace.png written.")
# map num_gen_sims sim pts from latent space thru flow_model to data space:
sim_pts = utils.generate_sim_pts(
    flow_model,
    run_params["num_gen_sims"],
    mean,
    cov,
    pca,
    regen_pts=mapped_training_pts,
)
# data space plot:
utils.plot_pts_2d(
    sim_pts,
    main_pts_label="mapped sim pts",
    side="data",
    plotfile=run_params["output_dir"] + "/test_output_dataspace.png",
)
print("test_output_dataspace.png written.")
