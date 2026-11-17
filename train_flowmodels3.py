"""Use case 3: generative classification (a) - cat and dog images.

The image-based generalization of train_flowmodels4.py, built on the image
handling of train_flowmodels2.py.  The flow is trained as an INN on labeled
images, x <-> [y, z]: forward, an image maps to a 2-component label vector y
(one-hot cat/dog) and a Gaussian latent z holding everything else about the
image; in reverse, a chosen label y* plus samples of z generate images of
that class.

Unlike use case 4 this is trained by maximum likelihood rather than MMD (MMD
is ineffective at the ~49,000 dimensions of these images): the y components
of the latent are Gaussian about the class's one-hot vector, the z components
are standard Gaussian.  That makes the model a class-conditional density
p(image | class), and classifying an image by which class mean its y lands
nearest is classification by Bayes' rule under the model.

Requires IMAGES_PATH with train/cat, train/dog, val/cat, val/dog subdirs
(e.g. the Kaggle animal-faces dataset as used by train_flowmodels2.py).
"""

import os
from pathlib import Path

import numpy as np
import warnings

from file_utils import get_data_generator
from model_training import default_training_sequence, load_param_overrides
import inn_utils
import utils

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these


def main():  # noqa: C901

    # Default user parameter settings:
    # Can change here for local non-Docker usage; or change in params/params3.json for Docker use
    # -----------------------
    run_params = {
        "output_dir": "output",
        "model_dir": "model/flowmodels3/catsdogs_128x128",
        "dataset": "catsdogs",
        "num_gen_sims": 10,  # number of new simulated images to generate per class
        "img2lat_chunk_size": 32,  # minibatch size when mapping images -> latents to avoid OOM
        "do_train": True,  # true = training, false = inference w existing model in model_dir
        "images_path": os.environ.get("IMAGES_PATH", "/storage/data/afhq"),
        "do_imgs_and_points": True,  # classify val images, generate scatterplots, sim images, etc
        "classes": ["cat", "dog"],  # image subdirs (within train/ and val/) used as the classes
        "num_eval_images": 1000,  # number of val (and train) images classified for metrics/plots
        "sample_temperature": 0.7,  # std of z when generating images (<1 trades variety for quality)
        "do_class_swap": True,  # regenerate some val images with their label vector swapped
        "num_swap_images": 6,  # number of val images used for the class-swap demo
    }
    training_params = {
        "num_epochs": 150,
        "batch_size": 64,  # max on g4dn.xlarge with current image size (fills memory)
        "reg_level": 1e-5,  # regularization level for the L2 reg in realNVP hidden layers
        "learning_rate": [4e-5, 13770, 0.85],  # decay 15% every ~30 epochs  # [initial_rate, decay_steps, decay_rate]
        # "learning_rate": 1e-4,  # scaler -> constant learning rate; vector of 3 -> lr schedule
        #     decayed_lr = initial_rate * decay_rate ^ (step / decay_steps)
        #     decay_steps = step * ln(decay_rate) / ln(decayed_lr / initial_rate)
        "early_stopping_patience": 40,  # value <=0 turns off early_stopping
        "num_data_input": 9800,  # num training images (cats + dogs in animal-faces train/)
        "augmentation_factor": 3,  # set >1 to have augmentation turned on
        "grad_norm_thresh": 75,  # if not None, clip norm of gradients at this thresh
        "log_scale_clip": 4,  # clip log-scale outputs to [-value, value]; <=0 disables
        "jit_compile": True,  # boolean, normally True but sometimes useful in debugging
        "tracking_tool": "mlflow",  # "tensorboard" or "mlflow"
        "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
        "tracking_expt_name": "flowmodels3",
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
        # INN (x <-> [y, z]) params:
        "y_dim": 2,  # label vector: one component per class (must equal len(classes))
        "lambda_nll": 49152.0,  # weight of the per-dimension likelihood term.  Set to
        #     prod(image_shape) = 128*128*3 so the loss is the total NLL per image, the same
        #     scale as train_flowmodels2.py's loss (so its learning rate and gradient
        #     clipping settings carry over).
        "y_sigma": 0.05,  # std of the label components about their one-hot class vector;
        #     small relative to the distance between class vectors so the classes separate
        "lambda_y": 1000.0,  # extra weight on (y_pred - y_true)^2: the label is 2 of 49152
        #     dimensions, so the likelihood alone gives it little pull; raise if accuracy is poor
        "lambda_z": 0.0,  # MMD terms off (ineffective at this dimensionality)
        "lambda_x": 0.0,
        "lambda_pad": 0.0,  # no padding
    }
    load_param_overrides(run_params, training_params, model_arch_params)
    os.makedirs(run_params["output_dir"], exist_ok=True)
    os.makedirs(run_params["model_dir"], exist_ok=True)
    # List the param settings:
    print("")
    utils.print_run_params(**run_params, **training_params, **model_arch_params)
    class_names = list(run_params["classes"])
    num_classes = len(class_names)
    if model_arch_params["y_dim"] != num_classes:
        raise ValueError("model_arch_params['y_dim'] must equal len(run_params['classes']).")
    class_colors = [f"C{k}" for k in range(num_classes)]
    class_means = np.eye(num_classes, dtype=np.float32)  # one-hot label vectors
    outdir = run_params["output_dir"]
    chunk = run_params["img2lat_chunk_size"]

    # Get the data
    # ------------
    # Same generators as train_flowmodels2.py, but class_mode="categorical"
    # makes them yield (images, one-hot labels) and shuffles the image order
    # so every batch mixes the classes.
    train_generator = get_data_generator(
        dataset=run_params["dataset"],
        batch_size=training_params["batch_size"],
        images_path=run_params["images_path"] + "/train",
        target_size=model_arch_params["image_shape"][:2],
        class_mode="categorical",
        classes=class_names,
    )
    val_generator = get_data_generator(
        dataset=run_params["dataset"],
        batch_size=training_params["batch_size"],
        images_path=run_params["images_path"] + "/val",
        target_size=model_arch_params["image_shape"][:2],
        class_mode="categorical",
        classes=class_names,
    )
    sample_images, sample_labels = next(train_generator)
    print(
        "train_generator test: shapes of one batch (images, labels): ",
        sample_images.shape, sample_labels.shape, "\n",
    )

    # Train the model
    # ---------------
    flow_model, history = default_training_sequence(
        train_generator, run_params, training_params, model_arch_params
    )

    # Analyze/plot various model results
    # ----------------------------------
    metrics = {}
    if run_params["do_imgs_and_points"]:
        num_eval = int(run_params["num_eval_images"])

        # Forward direction: map labeled images to label space and latent space.
        print(f"Now mapping {num_eval} val images to (label, latent) space...")
        val_images, val_onehot = inn_utils.collect_pairs(val_generator, num_eval)
        val_labels = np.argmax(val_onehot, axis=1)
        val_y, val_z, _ = flow_model.forward_yz(val_images, batch_size=chunk)
        val_pred = _nearest_class(val_y, class_means)
        metrics["val_accuracy"] = float(np.mean(val_pred == val_labels))
        metrics["val_label_mse"] = float(np.mean((val_y - val_onehot) ** 2))

        print(f"Now mapping {num_eval} training images to (label, latent) space...")
        train_images, train_onehot = inn_utils.collect_pairs(train_generator, num_eval)
        train_labels = np.argmax(train_onehot, axis=1)
        train_y, train_z, _ = flow_model.forward_yz(train_images, batch_size=chunk)
        train_pred = _nearest_class(train_y, class_means)
        metrics["train_accuracy"] = float(np.mean(train_pred == train_labels))
        del train_images
        print(
            f"Classification accuracy: train {metrics['train_accuracy']:.4f}, "
            f"val {metrics['val_accuracy']:.4f}"
        )

        # Quick latent stats to gauge match of z to N(0,I)
        latent_metrics = inn_utils.latent_stats(train_z)
        print("Latent stats on mapped training pts (z part):", latent_metrics)
        metrics.update(latent_metrics)

        print("Now plotting label space, confusion matrix, and 2D projection of latent points.")
        inn_utils.plot_labeled_pts_2d(
            val_y[:, :2],
            val_labels,
            class_colors,
            plotfile=outdir + "/val_points_labelspace.png",
            side="label",
            title="val images mapped to label (y) space, colored by true class",
            label_names=class_names,
            marker_pts=class_means[:, :2],
            axis_labels=(f"y[{class_names[0]}]", f"y[{class_names[1]}]"),
        )
        inn_utils.plot_confusion_matrix(
            inn_utils.confusion_matrix(val_labels, val_pred, num_classes),
            class_names,
            plotfile=outdir + "/val_confusion.png",
            title="Val images: predicted vs true class",
        )
        num_scatter = min(200, num_eval)
        utils.plot_pts_2d(
            train_z[:num_scatter],
            main_pts_label="mapped train pts (z)",
            side="latent",
            plotfile=outdir + "/training_points_latentspace.png",
            other_pts=val_z[:9],
            other_pts_label="val images",
        )
        print("val_points_labelspace.png, val_confusion.png, training_points_latentspace.png written.")

        # Reverse direction: for each class label y*, sample z and generate images.
        num_sims = int(run_params["num_gen_sims"])
        temperature = float(run_params["sample_temperature"])
        sim_rows = []
        for k, name in enumerate(class_names):
            print(f"Now generating {num_sims} simulated '{name}' images...")
            sim_rows.append(
                flow_model.sample_posterior(
                    class_means[k], num_sims, temperature=temperature, batch_size=5
                )
            )
        inn_utils.save_image_grid(
            sim_rows,
            plotfile=outdir + "/sim_images_by_class.png",
            row_labels=[f"y* = {name}" for name in class_names],
            title=f"Generated images x ~ p(x | class), temperature {temperature}",
        )
        print("sim_images_by_class.png written.")

        if run_params["do_class_swap"]:
            # Keep each image's z (everything but the class) and regenerate it
            # with its own label vector and with the other class's.
            num_swap = min(int(run_params["num_swap_images"]), num_eval)
            swap_rows = [val_images[:num_swap]]
            row_labels = ["original"]
            for k, name in enumerate(class_names):
                swap_rows.append(
                    flow_model.sample_posterior(
                        np.repeat(class_means[k : k + 1], num_swap, axis=0),
                        z=val_z[:num_swap],
                        y_noise_scale=0.0,
                        batch_size=5,
                    )
                )
                row_labels.append(f"same z, y* = {name}")
            inn_utils.save_image_grid(
                swap_rows,
                plotfile=outdir + "/class_swap_images.png",
                row_labels=row_labels,
                title="Val images regenerated from their latent z with each class label",
            )
            print("class_swap_images.png written.")
        print("Done.")

        print("Post-training metrics:")
        for key, value in metrics.items():
            print(f"    {key:<32} {value: .6f}")

    # Log analysis/plot results to MLflow (if specified)
    # --------------------------------------------------
    if training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open"):
        import mlflow
        if metrics:
            mlflow.log_metrics(metrics)
        for p in Path(run_params["output_dir"]).glob("*.png"):
            mlflow.log_artifact(str(p), artifact_path="plots")
        mlflow.log_artifact(run_params["model_dir"] + "/flow_model_summary.txt", artifact_path="model")
        mlflow.log_artifact(run_params["model_dir"] + "/model_arch.json", artifact_path="model")
        # (the model weights file at 2GB+ is too big to log in mlflow)
        mlflow.end_run()
        run_params["mlflow_run_open"] = False


def _nearest_class(y_pred, class_means):
    """Index of the class mean nearest each predicted label vector.  With
    equal class priors and equal y_sigma this is the model's Bayes classifier,
    since all other terms of log p(image | class) are the same for every
    class."""
    d = np.linalg.norm(y_pred[:, None, :] - class_means[None, :, :], axis=-1)
    return np.argmin(d, axis=1)


if __name__ == "__main__":
    main()
