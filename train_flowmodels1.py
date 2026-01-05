import warnings

import numpy as np

import utils
from file_utils import get_data_generator
from flow_model import default_training_sequence

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def _unwrap_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


run_params = {
    "output_dir": "output",  # local artifacts storage area before possibly logging to mlflow
    "model_dir": "models/flowmodels1",  # local model storage area before possibly logging to mlflow
    "dataset": "moons",  # "moons", "gmm", "mvn"
    "num_gen_sims": 1000,  # number of new simulated data to generate
    "do_train": True,  # true = training, false = inference w existing model in model_dir
}
training_params = {
    "num_epochs": 20,
    "batch_size": 256,
    "reg_level": 0.0,  # 0.01  # regularization level for the L2 reg in realNVP hidden layers
    "learning_rate": 0.0001,  # scaler -> constant learning rate; vector of 3 -> lr schedule
    # "learning_rate": [0.001, 300, 0.90],  # [initial_rate, decay_steps, decay_rate]
    #     decayed_lr = initial_rate * decay_rate ^ (step / decay_steps)
    #     decay_steps = step * ln(decay_rate) / ln(decayed_lr / initial_rate)
    "early_stopping_patience": 10,  # value <=0 turns off early_stopping
    # note current model arch has 534,544 params:
    "num_data_input": 50000,  # num training data pts or images (whether pts or files)
    "augmentation_factor": 1,  # set >1 to have augmentation turned on
    "grad_norm_thresh": None,  # if not None, clip norm of gradients at this thresh
    "jit_compile": True,  # boolean, normally True but sometimes useful in debugging
    "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
    "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
    "tracking_expt_name": "flowmodels1",
}
model_arch_params = {
    "image_shape": (2,),  # 2D points with (no color labels in this run)
    "bijector": "realnvp-based",
    "flow_steps": 8,  # number of realnvp-based affine coupling layers
    "hidden_layers": [256, 256],  # nodes/denselayer or filters/cnnlayer in affine coupling layers
    "validate_args": True,
}
# List the param settings:
print("")
utils.print_run_params(**run_params, **training_params, **model_arch_params)


# Get the data
# ------------
train_generator = get_data_generator(
    dataset=run_params["dataset"],
    batch_size=training_params["batch_size"],
)
sample_batch = _unwrap_batch(next(train_generator))
print("train_generator test: shape of one batch: ", sample_batch.shape, "\n")
if run_params["dataset"] in ["moons", "gmm", "mvn"]:
    # Quick sanity-check plot of some of the data for this group of 2D problems
    input_data_test = np.concatenate(
        [_unwrap_batch(next(train_generator)) for _ in range(20)], axis=0
    )
    datain_plot_path = run_params["output_dir"] + "/test_input_dataspace.png"
    utils.plot_pts_2d(
        input_data_test,
        main_pts_label="original train pts",
        side="data",
        plotfile=datain_plot_path,
    )
    print("test_input_dataspace.png written.")


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
mapped_training_pts, mean, cov, pca, top_outliers, closest_to_mean = (
    utils.imgs_to_gaussian_pts(flow_model, train_generator, 1000)
)
# latent space plot:
latent_plot_path = run_params["output_dir"] + "/test_output_latentspace.png"
utils.plot_pts_2d(
    mapped_training_pts,
    main_pts_label="mapped train pts",
    side="latent",
    plotfile=latent_plot_path,
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
dataout_plot_path = run_params["output_dir"] + "/test_output_dataspace.png"
utils.plot_pts_2d(
    sim_pts,
    main_pts_label="mapped sim pts",
    side="data",
    plotfile=dataout_plot_path,
)
print("test_output_dataspace.png written.")
if training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open"):
    import mlflow
    for artifact_path in [latent_plot_path, datain_plot_path, dataout_plot_path]:
        mlflow.log_artifact(artifact_path, artifact_path="plots")
    mlflow.log_artifact(run_params["model_dir"], artifact_path="model")
    mlflow.end_run()
    run_params["mlflow_run_open"] = False
