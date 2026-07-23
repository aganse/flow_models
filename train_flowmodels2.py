from pathlib import Path
import numpy as np
import utils
import warnings
import os

from file_utils import get_data_generator, image_files_to_data_generator
from flow_model import default_training_sequence

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def _unwrap_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


run_params = {
    "output_dir": "output",
    "model_dir": "model/flowmodels2/cats_128x128",
    "dataset": "cats",
    "num_gen_sims": 10,  # number of new simulated images to generate
    "img2lat_chunk_size": 32,  # minibatch size when mapping images -> latents to avoid OOM
    "do_train": True,  # true = training, false = inference w existing model in model_dir
    "images_path": "/storage/data/afhq",  # local filesys dir containing train/ and val/ subdirs
    # "images_path": "s3://mybucket",  # this works but is pretty slow, needs some optimizing
    "do_imgs_and_points": True,  # generate scatterplots, sim images, etc:  not dataset specific
    "do_interp": False,  # interp sim images between some training points:  cat dataset specific
    # Sampling-related knobs:
    "sampling_mode": "pca",  # "pca" (legacy) or "direct" (from N(0,I)) for sim generation
    "cov_scale": 0.25,  # multiplier on reduced_cov when sampling via PCA (legacy was 0.25)
    "pca_n_components": 200,  # legacy=100; None to skip PCA in stats/sampling; higher value uses more PCs
    "pca_solver": "randomized",  # "auto" or "randomized": "randomized" to scale PCA to larger component counts
    "regen_source": "flow_base",  # "pca_stats", "flow_base", or "train_pts" for sim images
}
training_params = {
    "num_epochs": 50,
    "batch_size": 64,  # max on g4dn.xlarge with current image size (fills memory)
    "reg_level": 1e-5,  # regularization level for the L2 reg in realNVP hidden layers
    # "learning_rate": 1e-4,  # scaler -> constant learning rate; vector of 3 -> lr schedule
    "learning_rate": [1e-4, 2370, 0.75],  # 7.5e-5 at 10th epoch  # [initial_rate, decay_steps, decay_rate]
    #     decayed_lr = initial_rate * decay_rate ^ (step / decay_steps)
    #     decay_steps = step * ln(decay_rate) / ln(decayed_lr / initial_rate)
    "early_stopping_patience": 30,  # value <=0 turns off early_stopping
    # note current model arch has 534,544 params:
    "num_data_input": 5100,  # num training data pts or images (whether pts or files)
    "augmentation_factor": 3,  # set >1 to have augmentation turned on
    "grad_norm_thresh": 25,  # if not None, clip norm of gradients at this thresh
    "log_scale_clip": 4,  # clip log-scale outputs to [-value, value]; <=0 disables
    "jit_compile": True,  # boolean, normally True but sometimes useful in debugging
    "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
    "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
    "tracking_expt_name": "flowmodels2",
    "save_model_weights": True,
}
model_arch_params = {
    "image_shape": (128, 128, 3),  # (height, width, channels) of images
    "bijector": "glow",  # "realnvp-based" or "glow"
    # realnvp-based params:
    "realnvp_flow_steps": 6,  # number of realnvp-based affine coupling layers
    "realnvp_hidden_layers": [512, 512],  # nodes/layer in realnvp-based affine coupling layers
    "realnvp_permutation": "alternating",  # "alternating" or "random" permutation between coupling layers
    # glow params:
    "glow_num_blocks": 3,  # number of multi-scale levels (need >=3 for 128x128)
    "glow_steps_per_block": 6,  # flow steps per level (paper uses 32; 8 is a lighter start)
    "glow_num_hidden": 128,  # filters in glow coupling CNN (paper uses 400; could use 256 here)
    "validate_args": False,
}
# List the param settings:
print("")
utils.print_run_params(**run_params, **training_params, **model_arch_params)
os.makedirs(run_params["output_dir"], exist_ok=True)
os.makedirs(run_params["model_dir"], exist_ok=True)


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
        utils.imgs_to_gaussian_pts(
            flow_model,
            train_generator,
            1000,
            chunk_size=run_params["img2lat_chunk_size"],
            neigvals=run_params["pca_n_components"],
            pca_solver=run_params["pca_solver"],
        )
    )
    # Quick latent stats to gauge match to N(0,I)
    latent_mean = np.mean(mapped_training_pts, axis=0)
    latent_std = np.std(mapped_training_pts, axis=0)
    print(
        "Latent stats on mapped training pts: "
        f"mean of means={latent_mean.mean():.4f}, "
        f"min/median/max mean=({latent_mean.min():.4f}, {np.median(latent_mean):.4f}, {latent_mean.max():.4f}); "
        f"mean std={latent_std.mean():.4f}, "
        f"min/median/max std=({latent_std.min():.4f}, {np.median(latent_std):.4f}, {latent_std.max():.4f})"
    )
    if training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open"):
        import mlflow
        mlflow.log_metrics({
            "latent_mean_of_means": float(latent_mean.mean()),
            "latent_median_mean":   float(np.median(latent_mean)),
            "latent_max_abs_mean":  float(np.abs(latent_mean).max()),
            "latent_mean_std":      float(latent_std.mean()),
            "latent_median_std":    float(np.median(latent_std)),
            "latent_min_std":       float(latent_std.min()),
            "latent_max_std":       float(latent_std.max()),
        })
    print("Now calculating Gaussian pts corresponding to first 9 'other' images...")
    other_pts, _, _, _, _, _ = utils.imgs_to_gaussian_pts(
        flow_model,
        other_generator,
        9,
        chunk_size=run_params["img2lat_chunk_size"],
        neigvals=run_params["pca_n_components"],
        pca_solver=run_params["pca_solver"],
    )
    effective_pca_dim = pca.n_components_ if pca is not None else None
    print(
        f"Sampling config: mode={run_params['sampling_mode']}, "
        f"regen_source={run_params['regen_source']}, "
        f"pca_components={effective_pca_dim}, "
        f"cov_scale={run_params['cov_scale']}"
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

    if top_outliers is None or closest_to_mean is None:
        print("PCA stats not available; skipping outlier/inlier regenerations.")
    else:
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
    sim_regen_pts = None
    sim_sampling_mode = run_params["sampling_mode"]
    if run_params["regen_source"] == "train_pts":
        sim_regen_pts = mapped_training_pts[14:]
    elif run_params["regen_source"] == "flow_base":
        sim_sampling_mode = "direct"
    elif run_params["regen_source"] != "pca_stats":
        print(
            f"Warning: unknown regen_source '{run_params['regen_source']}', "
            "defaulting to pca_stats."
        )
    if sim_sampling_mode == "pca" and pca is None:
        print("PCA stats not available; switching sampling_mode to 'direct'.")
        sim_sampling_mode = "direct"
    sim_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov,
        pca,
        filename=sim_images_path,
        batch_size=5,
        regen_pts=sim_regen_pts,
        sampling_mode=sim_sampling_mode,
        cov_scale=run_params["cov_scale"],
        add_plot_num=True,
    )
    # Quick latent stats to gauge match to N(0,I)
    sim_latent_mean = np.mean(sim_pts, axis=0)
    sim_latent_std = np.std(sim_pts, axis=0)
    print(
        "Latent stats on simulated pts: "
        f"mean of means={sim_latent_mean.mean():.4f}, "
        f"min/median/max mean=({sim_latent_mean.min():.4f}, {np.median(sim_latent_mean):.4f}, {sim_latent_mean.max():.4f}); "
        f"mean std={sim_latent_std.mean():.4f}, "
        f"min/median/max std=({sim_latent_std.min():.4f}, {np.median(sim_latent_std):.4f}, {sim_latent_std.max():.4f})"
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
    gaussian_points, _, _, _ = utils.imgs_to_gaussian_pts(
        flow_model,
        image_gen(),
        2,
        chunk_size=run_params["img2lat_chunk_size"],
    )
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

# Log to MLflow (if specified)
# ----------------------------
if training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open"):
    import mlflow
    for p in Path(run_params["output_dir"]).glob("*.png"):
        mlflow.log_artifact(str(p), artifact_path="plots")
    mlflow.log_artifact(run_params["model_dir"] + "/flow_model_summary.txt", artifact_path="model")
    mlflow.log_artifact(run_params["model_dir"] + "/model_arch.json", artifact_path="model")
    # (the model weights file at 2GB+ is too big to log in mlflow)
    mlflow.end_run()
    run_params["mlflow_run_open"] = False
