import warnings

import utils
from file_utils import get_data_generator, image_files_to_data_generator
from flow_model import default_training_sequence

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def _unwrap_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


run_params = {
    "output_dir": "output",
    "model_dir": "models/flowmodels2/cats_256x256new",
    "dataset": "cats",
    "num_gen_sims": 10,  # number of new simulated images to generate
    "do_train": True,  # true = training, false = inference w existing model in model_dir
    "images_path": "s3://aganse-images",  # (substitute your own s3 bucket name here)
    "do_imgs_and_points": True,  # generate scatterplots, sim images, etc:  not dataset specific
    "do_interp": False,  # interp sim images between some training points:  cat dataset specific
}
training_params = {
    "num_epochs": 20,
    "batch_size": 8,
    "reg_level": 0.0,  # 0.01  # regularization level for the L2 reg in realNVP hidden layers
    "learning_rate": 0.00001,  # scaler -> constant learning rate; vector of 3 -> lr schedule
    # "learning_rate": [0.001, 300, 0.90],  # [initial_rate, decay_steps, decay_rate]
    #     decayed_lr = initial_rate * decay_rate ^ (step / decay_steps)
    #     decay_steps = step * ln(decay_rate) / ln(decayed_lr / initial_rate)
    "early_stopping_patience": 10,  # value <=0 turns off early_stopping
    # note current model arch has 534,544 params:
    "num_data_input": 5000,  # num training data pts or images (whether pts or files)
    "augmentation_factor": 1,  # set >1 to have augmentation turned on
    "grad_norm_thresh": None,  # if not None, clip norm of gradients at this thresh
    "jit_compile": False,  # boolean, affects GPU runs: false=slower but won't
                           # crash on GPU runs (irrelevant on CPU-only runs)
    "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
    "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
    "tracking_expt_name": "flow_models2",
}
model_arch_params = {
    "image_shape": (256, 256, 3),  # (height, width, channels) of images
    "bijector": "realnvp-based",  # "realnvp-based" or "glow"
    "flow_steps": 6,  # number of realnvp-based affine coupling layers
    "hidden_layers": [512, 512],  # nodes/layer in realnvp-based affine coupling layers
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
    images_path=run_params["images_path"] + "/train",
    target_size=model_arch_params["image_shape"][:2],
)
other_generator = get_data_generator(
    dataset=run_params["dataset"],
    batch_size=training_params["batch_size"],
    images_path=run_params["images_path"] + "/val",
    target_size=model_arch_params["image_shape"][:2],
)
print("train_generator test: shape of one batch: ", next(train_generator).shape, "\n")


# Train the model
# ---------------
flow_model, history = default_training_sequence(
    train_generator, run_params, training_params, model_arch_params
)


# Analyze/plot various model results
# ----------------------------------
if run_params["do_imgs_and_points"]:
    # note that training_pts, mean, cov here are all high-dimensional objects.
    # map 1000 pts from train_generator thru flow_model to latent space:
    print("Now calculating Gaussian pts corresponding to first 1000 training images...")
    mapped_training_pts, mean, reduced_cov, pca, top_outliers, closest_to_mean = (
        utils.imgs_to_gaussian_pts(flow_model, train_generator, 1000)
    )
    print("Now calculating Gaussian pts corresponding to first 9 'other' images...")
    other_pts, _, _, _, _, _ = utils.imgs_to_gaussian_pts(
        flow_model, other_generator, 9
    )
    print("Now plotting 2D projection of those training points.")
    trainpts_latent_plot_path = run_params["output_dir"] + "/training_points_latentspace.png"
    utils.plot_pts_2d(
        mapped_training_pts,
        main_pts_label="mapped train pts",
        side="latent",
        plotfile=trainpts_latent_plot_path,
        mean=mean,
        sim_pts=top_outliers,
        sim_pts_label="top outliers",
        other_pts=closest_to_mean,
        other_pts_label="close to mean",
        num_regen=5,
    )
    print("training_points_latentspace.png written.")

    print(f"Now regenerating {run_params['num_gen_sims']} outlier images...")
    outliers_images_path = run_params["output_dir"] + "/outlier_image"
    outlier_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov,
        pca,
        filename=outliers_images_path,
        batch_size=5,
        regen_pts=top_outliers,
        add_plot_num=True,
    )
    print(f"Now regenerating {run_params['num_gen_sims']} inlier images...")
    inliers_images_path = run_params["output_dir"] + "/inlier_image"
    inlier_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov,
        pca,
        filename=inliers_images_path,
        batch_size=5,
        regen_pts=closest_to_mean,
        add_plot_num=True,
    )
    print(f"Now regenerating {run_params['num_gen_sims']} training images...")
    regen_images_path = run_params["output_dir"] + "/regen_image"
    regen_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov,
        pca,
        filename=regen_images_path,
        batch_size=5,
        regen_pts=mapped_training_pts[14:],
        add_plot_num=True,
    )
    print(f"Now generating {run_params['num_gen_sims']} simulated images...")
    sim_images_path = run_params["output_dir"] + "/sim_image"
    sim_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov / 4,
        pca,
        filename=sim_images_path,
        batch_size=5,
        add_plot_num=True,
    )
    print("Now plotting 2D projection of training+sim+other points.")
    compare_images_path = run_params["output_dir"] + "/compare_points_2d.png"
    utils.plot_pts_2d(
        mapped_training_pts,
        plotfile=compare_images_path,
        mean=mean,
        sim_pts=sim_pts,
        other_pts=other_pts,
        num_regen=5,
    )
    print("Done.")


if run_params["do_interp"]:
    # experimenting with interpolating images between a pair of points in latent space:
    white_cat = "data/afhq/val/cat/flickr_cat_000016.jpg"
    calico_cat = "data/afhq/val/cat/flickr_cat_000056.jpg"
    gray_cat = "data/afhq/val/cat/flickr_cat_000076.jpg"
    pug_dog = "data/afhq/val/dog/flickr_dog_000079.jpg"
    white_pitbull_dog = "data/afhq/val/dog/flickr_dog_000054.jpg"
    sheltie_dog = "data/afhq/val/dog/flickr_dog_000334.jpg"  # tan & blk
    tiger = "data/afhq/val/wild/flickr_wild_001043.jpg"
    lion = "data/afhq/val/wild/flickr_wild_001397.jpg"

    filenames = [white_cat, gray_cat]
    image_gen = image_files_to_data_generator(
        filenames, target_size=model_arch_params["image_shape"][:2]
    )
    gaussian_points, _, _, _ = utils.imgs_to_gaussian_pts(flow_model, image_gen(), 2)
    print(gaussian_points.shape)
    print(gaussian_points)
    gaussian_points = utils.interpolate_between_points(
        gaussian_points, 4, path="euclidean"
    )
    interp_images_path = run_params["output_dir"] + "/interp_image"
    _ = utils.generate_imgs_in_batches(
        flow_model,
        4,
        None,
        None,
        None,
        filename=interp_images_path,
        batch_size=4,
        regen=gaussian_points,
    )

if training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open"):
    import mlflow

    plots_list = [
        trainpts_latent_plot_path, outliers_images_path, inliers_images_path,
        regen_images_path, sim_images_path, compare_images_path
    ]
    if run_params["do_interp"]:
        plots_list.append(interp_images_path)
    for artifact_path in plots_list:
        mlflow.log_artifact(artifact_path, artifact_path="plots")
    mlflow.end_run()
    run_params["mlflow_run_open"] = False
