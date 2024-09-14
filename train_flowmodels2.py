import warnings

# Can use this for local filesystem in place of S3ImageDataGenerator for s3:
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils
from file_utils import S3ImageDataGenerator
from flow_model import default_training_sequence


warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these

run_params = {
    "output_dir": "output",
    "model_dir": "models/flowmodels2/cats_256x256new",
    "do_train": True,  # true = training, false = inference w existing model in model_dir
    "use_tensorboard": True,
    "data": "moons",  # "moons", "GMM", "cats", "catsdogs", "invkin", "glacgrav"
    "num_gen_sims": 10,  # number of new simulated images to generate
    "do_imgs_and_points": True,  # generate scatterplots, sim images, etc:  not dataset specific
    "do_interp": False,  # interp sim images between some training points:  cat dataset specific
}
training_params = {
    "num_epochs": 10,
    "batch_size": 128,
    "reg_level": 0,  # 0.01  # regularization level for the L2 reg in realNVP hidden layers
    "learning_rate": 0.00001,  # scaler -> constant rate; list-of-3 -> exponential decay
    # "learning_rate": [0.001, 500, 0.95],  # [initial_rate, decay_steps, decay_rate]
    "early_stopping_patience": 0,  # value <=0 turns off early_stopping
    "num_data_input": 5600,  # num training data pts or images (whether pts or files)
    "augmentation_factor": 2,  # set >1 to have augmentation turned on
}
model_arch_params = {
    "image_shape": (256, 256, 3),  # (height, width, channels) of images
    "hidden_layers": [512, 512],  # nodes per layer within affine coupling layers
    "flow_steps": 6,  # number of affine coupling layers
    "validate_args": True,
}
# List the param settings:
utils.print_run_params(**run_params, **training_params, **model_arch_params)


# datagen = ImageDataGenerator(  # images in local filesystem
datagen = S3ImageDataGenerator(  # images in S3 bucket
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.0,  # 0.1,  # still debugging this feature for S3ImageDG
    rotation_range=10,
    width_shift_range=0.0,  # 0.1,  # still debugging this feature for S3ImageDG
    height_shift_range=0.0,  # 0.1,  # still debugging this feature for S3ImageDG
)
train_generator = datagen.flow_from_directory(
    "s3://mybucket/train",
    target_size=model_arch_params["image_shape"][:2],  # images get resized to this size
    batch_size=training_params["batch_size"],
    class_mode=None,  # unsupervised learning so no class labels
    shuffle=False,  # possibly helpful for training but pain for plot revamps/additions
)
other_generator = datagen.flow_from_directory(
    "s3://mybucket/val",
    target_size=model_arch_params["image_shape"][:2],  # images get resized to this size
    batch_size=training_params["batch_size"],
    class_mode=None,  # unsupervised learning so no class labels
)

print("train_generator test: shape of one batch: ", next(train_generator).shape, "\n")


flow_model = default_training_sequence(
    train_generator, run_params, training_params, model_arch_params
)


if run_params["do_imgs_and_points"]:
    # Note that training_pts, mean, cov here are all high-dimensional objects:
    N = 1000
    print(
        f"Now calculating Gaussian pts corresponding to first {N} training images...",
        flush=True,
    )
    training_pts, mean, reduced_cov, pca, top_outliers, closest_to_mean = (
        utils.imgs_to_gaussian_pts(flow_model, train_generator, N)
    )
    print(
        "Now calculating Gaussian pts corresponding to first 9 'other' images...",
        flush=True,
    )
    other_pts, _, _, _, _, _ = utils.imgs_to_gaussian_pts(
        flow_model, other_generator, 9
    )
    print("Now plotting 2D projection of those training points.", flush=True)
    utils.plot_gaussian_pts_2d(
        training_pts,
        plotfile=run_params["output_dir"] + "/training_points.png",
        mean=mean,
        sim_pts=top_outliers,
        sim_pts_label="top outliers",
        other_pts=closest_to_mean,
        other_pts_label="close to mean",
        num_regen=5,
    )
    print(
        f"Now regenerating {run_params['num_gen_sims']} outlier images...", flush=True
    )
    outlier_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov,
        pca,
        filename=run_params["output_dir"] + "/outlier_image",
        batch_size=5,
        regen_pts=top_outliers,
        add_plot_num=True,
    )
    print(f"Now regenerating {run_params['num_gen_sims']} inlier images...", flush=True)
    inlier_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov,
        pca,
        filename=training_params["output_dir"] + "/inlier_image",
        batch_size=5,
        regen_pts=closest_to_mean,
        add_plot_num=True,
    )
    print(
        f"Now regenerating {run_params['num_gen_sims']} training images...",
        flush=True,
    )
    regen_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov,
        pca,
        filename=run_params["output_dir"] + "/regen_image",
        batch_size=5,
        regen_pts=training_pts[14:],
        add_plot_num=True,
    )
    print(
        f"Now generating {run_params['num_gen_sims']} simulated images...", flush=True
    )
    sim_pts = utils.generate_imgs_in_batches(
        flow_model,
        run_params["num_gen_sims"],
        mean,
        reduced_cov / 4,
        pca,
        filename=run_params["output_dir"] + "/sim_image",
        batch_size=5,
        add_plot_num=True,
    )
    print("Now plotting 2D projection of training+sim+other points.", flush=True)
    utils.plot_gaussian_pts_2d(
        training_pts,
        plotfile=run_params["output_dir"] + "/compare_points_2d.png",
        mean=mean,
        sim_pts=sim_pts,
        other_pts=other_pts,
        num_regen=5,
    )
    print("Done.", flush=True)


if run_params["do_interp"]:
    # Experimenting with interpolating images between a pair of points in latent space:
    white_cat = "data/afhq/val/cat/flickr_cat_000016.jpg"
    calico_cat = "data/afhq/val/cat/flickr_cat_000056.jpg"
    gray_cat = "data/afhq/val/cat/flickr_cat_000076.jpg"
    pug_dog = "data/afhq/val/dog/flickr_dog_000079.jpg"
    white_pitbull_dog = "data/afhq/val/dog/flickr_dog_000054.jpg"
    sheltie_dog = "data/afhq/val/dog/flickr_dog_000334.jpg"  # tan & blk
    tiger = "data/afhq/val/wild/flickr_wild_001043.jpg"
    lion = "data/afhq/val/wild/flickr_wild_001397.jpg"

    filenames = [white_cat, gray_cat]
    image_gen = utils.image_data_generator(
        filenames, target_size=model_arch_params["image_shape"][:2]
    )
    gaussian_points, _, _, _ = utils.imgs_to_gaussian_pts(flow_model, image_gen(), 2)
    print(gaussian_points.shape)
    print(gaussian_points)
    gaussian_points = utils.interpolate_between_points(
        gaussian_points, 4, path="euclidean"
    )
    _ = utils.generate_imgs_in_batches(
        flow_model,
        4,
        None,
        None,
        None,
        filename=run_params["output_dir"] + "/gen_image",
        batch_size=4,
        regen=gaussian_points,
    )
