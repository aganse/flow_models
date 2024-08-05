from datetime import datetime
import warnings

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.keras.callbacks import TensorBoard

from flow_model import FlowModel
import utils

warnings.filterwarnings("ignore", category=UserWarning)  # TFP spews a number of these

# Useful stuff when debugging but annoying otherwise:
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Print out the device each operation is assigned to, helping identify if any
# operations are unexpectedly running on the CPU. (caution: debug only: highly verbose!)
# tf.debugging.set_log_device_placement(True)

### Run params: ###
output_dir = "output"
model_dir = "models/cat_models/cats_128x128new"
do_train = True  # true = training, false = inference w existing model in model_dir
use_tensorboard = True
do_imgs_and_points = True  # generate scatterplots, sim images, etc:  not dataset specific
do_interp = False  # interp sim images between some training points:  cat dataset specific
### Training params: ###
num_epochs = 30
batch_size = 64
reg_level = 0  # 0.01  # regularization level for the L2 reg in realNVP hidden layers
learning_rate = 0.0001  # scaler -> constant rate; list-of-3 -> exponential decay
# learning_rate = [0.001, 500, 0.95]  # [initial_rate, decay_steps, decay_rate]
early_stopping_patience = 5  # value <=0 turns off early_stopping
num_image_files = 5600  # num training images (todo: auto-find from directory)
augmentation_factor = 4  # set >1 to have augmentation turned on
steps_per_epoch = num_image_files // batch_size * augmentation_factor
num_gen_images = 10  # number of new images to generate
### Model architecture params: ###
image_shape = (128, 128, 3)  # (height, width, channels) of images
hidden_layers = [512, 512]  # nodes per layer within affine coupling layers
flow_steps = 6  # number of affine coupling layers
validate_args = True
# Record those param settings:
utils.print_run_params(
    output_dir=output_dir, model_dir=model_dir, do_train=do_train,
    num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
    early_stopping_patience=early_stopping_patience, num_gen_images=num_gen_images,
    num_image_files=num_image_files, augmentation_factor=augmentation_factor,
    steps_per_epoch=steps_per_epoch, image_shape=image_shape,
    hidden_layers=hidden_layers, flow_steps=flow_steps,
    validate_args=validate_args
)


datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
train_generator = datagen.flow_from_directory(
    "data/train",
    target_size=image_shape[:2],  # images get resized to this size
    batch_size=batch_size,
    class_mode=None,  # unsupervised learning so no class labels
    shuffle=False,  # possibly helpful for training but pain for plot revamps/additions
)
other_generator = datagen.flow_from_directory(
    "data/anom",
    target_size=image_shape[:2],  # images get resized to this size
    batch_size=batch_size,
    class_mode=None,  # unsupervised learning so no class labels
)


flow_model = FlowModel(image_shape, hidden_layers, flow_steps, reg_level, validate_args)
print("")
flow_model.build(input_shape=(None, *image_shape))  # only necessary for .summary() before train
print("Still working on why model layer specs not outputting to model summary below...")
flow_model.summary()
# _ = model(X)
# model.summary()

if do_train:
    print("Training model...", flush=True)

    if isinstance(learning_rate, float):
        lrate = learning_rate
    elif isinstance(learning_rate, list) and len(learning_rate) == 3:
        lrate = ExponentialDecay(
            learning_rate[0],
            decay_steps=learning_rate[1],
            decay_rate=learning_rate[2],
            staircase=True
        )
    else:
        print("train.py: error: learning_rate not scalar or list of length 3.")
        quit()

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor="neg_log_likelihood",
            patience=early_stopping_patience,
            restore_best_weights=True
        ))
    if use_tensorboard:
        log_dir = f"./logs/train/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=False
        ))
    flow_model.compile(optimizer=Adam(learning_rate=lrate))
    infinite_train_generator = utils.infinite_generator(train_generator)
    flow_model.fit(
        infinite_train_generator,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )
    print("Done training model.", flush=True)
    flow_model.save_weights(model_dir + "/model_weights")
    print("Model weights saved to file.", flush=True)
else:
    print(f"Loading model weights from file in {model_dir}.", flush=True)
    flow_model.load_weights(model_dir + "/model_weights")


if do_imgs_and_points:
    # Note that training_pts, mean, cov here are all high-dimensional objects:
    N = 1000
    print(f"Now calculating Gaussian pts corresponding to first {N} training images...", flush=True)
    training_pts, mean, reduced_cov, pca, top_outliers, closest_to_mean = utils.imgs_to_gaussian_pts(flow_model, train_generator, N)
    print("Now calculating Gaussian pts corresponding to first 9 'other' images...", flush=True)
    other_pts, _, _, _, _, _ = utils.imgs_to_gaussian_pts(flow_model, other_generator, 9)
    print("Now plotting 2D projection of those training points.", flush=True)
    utils.plot_gaussian_pts_2d(training_pts, plotfile=output_dir + "/training_points.png", mean=mean, sim_pts=top_outliers, sim_pts_label="top outliers", other_pts=closest_to_mean, other_pts_label="close to mean", num_regen=5)
    print(f"Now regenerating {num_gen_images} outlier images...", flush=True)
    outlier_pts = utils.generate_imgs_in_batches(flow_model, num_gen_images, mean, reduced_cov, pca, filename=output_dir + "/outlier_image", batch_size=5, regen_pts=top_outliers, add_plot_num=True)
    print(f"Now regenerating {num_gen_images} inlier images...", flush=True)
    inlier_pts = utils.generate_imgs_in_batches(flow_model, num_gen_images, mean, reduced_cov, pca, filename=output_dir + "/inlier_image", batch_size=5, regen_pts=closest_to_mean, add_plot_num=True)
    print(f"Now regenerating {num_gen_images} training images...", flush=True)
    regen_pts = utils.generate_imgs_in_batches(flow_model, num_gen_images, mean, reduced_cov, pca, filename=output_dir + "/regen_image", batch_size=5, regen_pts=training_pts[14:], add_plot_num=True)
    print(f"Now generating {num_gen_images} simulated images...", flush=True)
    sim_pts = utils.generate_imgs_in_batches(flow_model, num_gen_images, mean, reduced_cov / 4, pca, filename=output_dir + "/sim_image", batch_size=5, add_plot_num=True)
    print("Now plotting 2D projection of training+sim+other points.", flush=True)
    utils.plot_gaussian_pts_2d(training_pts, plotfile=output_dir + "/compare_points_2d.png", mean=mean, sim_pts=sim_pts, other_pts=other_pts, num_regen=5)
    print("Done.", flush=True)


if do_interp:
    # Experimenting with interpolating images between a pair of points in latent space:
    white_cat = 'data/afhq/val/cat/flickr_cat_000016.jpg'
    calico_cat = 'data/afhq/val/cat/flickr_cat_000056.jpg'
    gray_cat = 'data/afhq/val/cat/flickr_cat_000076.jpg'
    pug_dog = 'data/afhq/val/dog/flickr_dog_000079.jpg'
    white_pitbull_dog = 'data/afhq/val/dog/flickr_dog_000054.jpg'
    sheltie_dog = 'data/afhq/val/dog/flickr_dog_000334.jpg'  # tan & blk
    tiger = 'data/afhq/val/wild/flickr_wild_001043.jpg'
    lion = 'data/afhq/val/wild/flickr_wild_001397.jpg'

    filenames = [white_cat, gray_cat]
    image_gen = utils.image_data_generator(filenames, target_size=image_shape[:2])
    gaussian_points, _, _, _ = utils.imgs_to_gaussian_pts(flow_model, image_gen(), 2)
    print(gaussian_points.shape)
    print(gaussian_points)
    gaussian_points = utils.interpolate_between_points(gaussian_points, 4, path='euclidean')
    _ = utils.generate_imgs_in_batches(flow_model, 4, None, None, None, filename=output_dir + "/gen_image", batch_size=4, regen=gaussian_points)
