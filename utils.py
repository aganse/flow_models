import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pprint
from scipy.spatial import distance
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf


def imgs_to_gaussian_pts(model, image_generator, N, neigvals=100, p_outliers=10):
    """Map input images (from data generator) through the model to points in the
    Gaussian latent space.  Also computes latent space stats in reduced coords
    (via pca, since too high dimensionality for later sampling).  Also computes
    the #p_outliers most extreme outliers (re euclidean dist) and #p_outliers
    points closest to the mean.

    Images coming out of image_generator are (MxM) pixels.
    N = number of images to draw from image_generator to map.
    Make sure neigvals<<M^2.
    """

    M = np.prod(next(image_generator)[0].shape[1:])
    neigvals = min(M, N, 100)
    # print("imgs_to_gaussian_pts: M=", M, ", N=", N)

    def get_n_images(data_generator, n):
        images = []
        while len(images) < n:
            img_batch = next(data_generator)[0]
            for img in img_batch:
                images.append(img)
                if len(images) == n:
                    images_np = np.array(images)
                    images_tf = tf.convert_to_tensor(images_np, dtype=tf.float32)
                    return images_tf

    images = get_n_images(image_generator, N)
    print("images.shape 1:", images.shape)
    images = tf.reshape(images, (-1, np.prod(images.shape[1:])))
    print("images.shape 2:", images.shape)
    gaussian_points = model.call(images)
    print("first several gaussian pts:", gaussian_points[:8])
    print("gpts.shape 3:", gaussian_points.shape)
    gaussian_points = gaussian_points.numpy()

    if N > 10:
        pca = PCA(n_components=neigvals)
        reduced_data = pca.fit_transform(gaussian_points)
        mean_full = np.mean(gaussian_points, axis=0)
        mean_reduced = np.mean(reduced_data, axis=0)
        cov_reduced = np.cov(reduced_data, rowvar=False)
        dists_reduced = np.array(
            [distance.euclidean(point, mean_reduced) for point in reduced_data]
        )
        outlier_indices = np.argsort(dists_reduced)[-p_outliers:]
        top_outliers = gaussian_points[outlier_indices]
        inlier_indices = np.argsort(dists_reduced)[:p_outliers]
        closest_to_mean = gaussian_points[inlier_indices]

    else:
        mean_full = None
        cov_reduced = None
        pca = None
        top_outliers = None
        closest_to_mean = None

    return gaussian_points, mean_full, cov_reduced, pca, top_outliers, closest_to_mean


def plot_pts_2d(
    train_pts,
    plotfile="compare_points_2d.png",
    mean=None,
    main_pts_label="train images",
    sim_pts=None,
    sim_pts_label="sim images",
    other_pts=None,
    other_pts_label="anomaly images",
    num_regen=None,
    side="latent",  # "latent" or "data"
):
    """Scatterplot of various categories of points in the Gaussian latent space."""

    print("plot_pts_2d train_pts.shape:", train_pts.shape)
    if train_pts.shape[1] > 2:
        print("applying PCA to reduce dimensions to 2 to plot...")
        pca = PCA(n_components=2)
        if sim_pts is not None and other_pts is not None:
            # Combine and find pca of combo plus origin(mean)
            mix_pts = pca.fit_transform(np.concatenate([sim_pts, other_pts, [mean]]))
            sim_pts = pca.transform(sim_pts)
            other_pts = pca.transform(other_pts)
            train_pts = pca.transform(train_pts)
        elif sim_pts is not None and other_pts is None:
            # Find pca of sim_pts plus origin(mean)
            mix_pts = pca.fit_transform(np.concatenate([sim_pts, [mean]]))
            sim_pts = pca.transform(sim_pts)
            train_pts = pca.transform(train_pts)
        elif sim_pts is None and other_pts is not None:
            # Find pca of other_pts plus origin(mean)
            mix_pts = pca.fit_transform(np.concatenate([other_pts, [mean]]))
            other_pts = pca.transform(other_pts)
            train_pts = pca.transform(train_pts)
        elif sim_pts is None and other_pts is None:
            # Find pca of just the training pts samples
            train_pts = pca.fit_transform(train_pts)

    fig, ax = plt.subplots()
    ax.scatter(
        train_pts[:, 0], train_pts[:, 1], color="C0", alpha=0.5, label=main_pts_label
    )

    if num_regen is not None:
        ax.scatter(
            train_pts[:num_regen, 0],
            train_pts[:num_regen, 1],
            color="cyan",
            label="regen images",
        )

    if sim_pts is not None:
        ax.scatter(sim_pts[:, 0], sim_pts[:, 1], color="C1", label=sim_pts_label)
        for i in range(sim_pts.shape[0]):
            ax.annotate(
                str(i + 1),
                (sim_pts[i, 0], sim_pts[i, 1]),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
                va="center",
            )

    if other_pts is not None:
        print(f"2D coordinates of the {other_pts_label} points in the scatterplot:")
        print(other_pts)
        ax.scatter(
            other_pts[:, 0], other_pts[:, 1], color="chartreuse", label=other_pts_label
        )
        for i in range(other_pts.shape[0]):
            ax.annotate(
                str(i + 1),
                (other_pts[i, 0], other_pts[i, 1]),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
                va="center",
            )

    # put axis outside plot on right side:
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if train_pts.shape[1] > 2:
        plt.title(f"2D PCA of points in {side} space")
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
    else:
        plt.title(f"2D points in {side} space")
        plt.xlabel("x")
        plt.ylabel("y")

    plt.savefig(plotfile, bbox_inches="tight")


def plot_gaussian_pts_1d(
    training_pts,
    plotfile="compare_points_1d.png",
    mean=None,
    reduced_cov=None,
    sim_pts=None,
    other_pts=None,
    other_pts_label="anomaly images",
    num_regen=None,
):
    """Histogram of the magnitudes of the (high-dimensional) gaussian vectors in
    the latent space.  Since those are normally distributed (by construction),
    if there's a reasonable number of samples then this histogram should look
    approximately gaussian as well (ie chi2 with large dof).
    """

    y_level = 0.01  # arbitrary height to plot individual comparison points at
    training_pts_1d = np.linalg.norm(training_pts, axis=1)  # compute vector magnitudes

    fig, ax = plt.subplots()
    sns.kdeplot(training_pts_1d, label="train images", ax=ax)

    if num_regen is not None:
        y_values = y_level * np.ones(num_regen)  # Create an array of y values
        ax.scatter(
            training_pts_1d[:num_regen],
            y_values,
            label="regen images",
            facecolors="C0",
            alpha=0.5,
            edgecolors="k",
        )

    if sim_pts is not None:
        y_values = y_level * np.ones(len(sim_pts))  # Create an array of y values
        ax.scatter(
            training_pts_1d[: len(sim_pts)], y_values, color="C1", label="sim images"
        )

    if other_pts is not None:
        y_values = y_level * np.ones(len(other_pts))  # Create an array of y values
        ax.scatter(
            training_pts_1d[: len(other_pts)],
            y_values,
            color="chartreuse",
            label=other_pts_label,
        )

    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5)
    )  # put axis outside plot on right side
    plt.title("1D distribution of mapped gaussian points")
    plt.xlabel("Gaussian vector magnitudes")
    plt.ylabel("Density and example points")
    plt.savefig(plotfile, bbox_inches="tight")


def generate_multivariate_normal_samples(mean, reduced_cov, pca, num_samples):
    """Used by generate_imgs_in_batches().  The high dimensionality requires
    generating samples in reduced space (via pca, hence reduced_cov), and then
    transforming back out to full dimension, and thus this function.
    """

    # Generate new samples in reduced space
    new_samples_reduced = np.random.multivariate_normal(
        # (make 1D mean into 2D, then rotate/reduce it, then put back to 1D)
        mean=np.squeeze(pca.transform([mean])),
        cov=reduced_cov,
        size=num_samples,
    )

    # Transform new samples back to original space
    new_samples = pca.inverse_transform(new_samples_reduced)
    new_samples_tf = tf.convert_to_tensor(new_samples, dtype=tf.float32)

    return new_samples_tf


def generate_imgs_in_batches(
    model,
    num_gen_images,
    mean,
    reduced_cov,
    pca,
    filename="sim_image",
    batch_size=10,
    regen_pts=None,
    add_plot_num=False,
):
    """Given latent space distribution params, and/or list of points to use
    (regen_pts), map those through the model into images.

    model: trained FlowModel object
    num_gen_images: number of points to map (ie samples to generate from
                    mean/reduced_cov or to draw from regen_pts)
    The following 3 come out of imgs_to_gaussian_pts():
      mean: numpy array of the full-dimensional vector mean point (ideally near 0)
      reduced_cov: the cov matrix computed in the reduced space from pca
      pca: the pca object from imgs_to_gaussian_pts()
    filename: string that numbers appended to for filenames of generated images
    batch_size: integer - note this is batches of generated images, not training data batches!
    regen_pts: (optional) numpy array of training_pts for regenerating images for
           first N of them, instead of generating random pts from mean & cov.
           Technically doesn't have to be training_pts, could be any array of pts.
    add_plot_num: boolean: add little orange id # at top left of output images
        to match them up to the numbers in the scatterplots.
    """

    num_batches = (num_gen_images + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        # Determine the number of images to generate in this batch
        current_batch_size = min(batch_size, num_gen_images - batch_idx * batch_size)

        if regen_pts is None:
            # Generate a batch of Gaussian samples using TensorFlow
            samples_tf = generate_multivariate_normal_samples(
                mean, reduced_cov, pca, current_batch_size
            )
        else:
            # Get next batch worth of points from supplied training_points
            regen_tf = tf.convert_to_tensor(regen_pts, dtype=tf.float32)
            samples_tf = regen_tf[
                (batch_idx * batch_size) : (batch_idx * batch_size + current_batch_size)
            ]

        for i in range(current_batch_size):
            # Map back through the invertible network
            generated_image = model.inverse(samples_tf[i : i + 1])
            generated_image = tf.reshape(generated_image, model.image_shape)

            # Save the generated image
            img = generated_image.numpy()
            img = (img * 255).astype(np.uint8)  # Convert back to uint8 format
            img_idx = batch_idx * batch_size + i + 1
            if add_plot_num:
                img = add_text_to_image(
                    img, str(img_idx), font_size=20, color="orange", bold=True
                )
            plt.imsave(f"{filename}_{img_idx}.png", img)

        print(
            f"Generated and saved {batch_idx * batch_size + current_batch_size} images out of {num_gen_images}"
        )

    return samples_tf


def generate_sim_pts(
    model,
    num_gen_images,
    mean,
    reduced_cov,
    pca,
    regen_pts=None,
    batch_size=10,
):
    """Given latent space distribution params, and/or list of points to use
    (regen_pts), map those through the model into images.

    model: trained FlowModel object
    num_gen_images: number of points to map (ie samples to generate from
                    mean/reduced_cov or to draw from regen_pts)
    The following 3 come out of imgs_to_gaussian_pts():
      mean: numpy array of the full-dimensional vector mean point (ideally near 0)
      reduced_cov: the cov matrix computed in the reduced space from pca
      pca: the pca object from imgs_to_gaussian_pts()
    filename: string that numbers appended to for filenames of generated images
    batch_size: integer - note this is batches of generated images, not training data batches!
    regen_pts: (optional) numpy array of training_pts for regenerating images for
           first N of them, instead of generating random pts from mean & cov.
           Technically doesn't have to be training_pts, could be any array of pts.
    add_plot_num: boolean: add little orange id # at top left of output images
        to match them up to the numbers in the scatterplots.
    """

    num_batches = (num_gen_images + batch_size - 1) // batch_size

    mapped_pts = []
    for batch_idx in range(num_batches):
        # Determine the number of images to generate in this batch
        current_batch_size = min(batch_size, num_gen_images - batch_idx * batch_size)

        if regen_pts is None:
            # Generate a batch of Gaussian samples using TensorFlow
            samples_tf = generate_multivariate_normal_samples(
                mean, reduced_cov, pca, current_batch_size
            )
        else:
            # Get next batch worth of points from supplied training_points
            regen_tf = tf.convert_to_tensor(regen_pts, dtype=tf.float32)
            samples_tf = regen_tf[
                (batch_idx * batch_size) : (batch_idx * batch_size + current_batch_size)
            ]

        for i in range(current_batch_size):
            # Map back through the invertible network
            generated_image = model.inverse(samples_tf[i : i + 1])
            generated_image = tf.reshape(generated_image, model.image_shape)
            mapped_pts.append(generated_image.numpy())

    sim_pts = np.array(mapped_pts)

    return sim_pts


def add_text_to_image(image, text, font_size, color, bold):
    """Annotate little plot-number in corner of images.
    For use as a option in generate_imgs_in_batches().
    """
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    # Set font weight
    # font = ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", font_size)
    font = ImageFont.load_default()

    # Add text
    draw.text((10, 10), text, font=font, fill=color)

    return np.array(img_pil)


def print_run_params(**kwargs):
    """Generic function to dump the args list into text file, for debug/logging."""

    if "output_dir" not in kwargs:
        raise ValueError(
            "print_run_params: error: 'output_dir' must be one of the kwargs."
        )

    print("Run params:", kwargs)

    output_dir = kwargs.pop("output_dir")
    file_path = output_dir + "/run_parameters.txt"
    with open(file_path, "w") as file:
        file.write("Parameters used in this run:\n")
        file.write(pprint.pformat(kwargs))

    print("")


def print_model_summary(model):
    """Just experimenting with alternate model summaries than model.summary()"""

    # Print the header
    print(f"{'Layer Name':<20}{'Output Shape':<20}{'#Parameters':<11}")
    print("-" * 51)

    # Print the layer details
    for layer in model.layers:
        layer_name = layer.name
        output_shape = str(layer.output_shape)
        num_params = layer.count_params()
        print(f"{layer_name:<20}{output_shape:<20}{num_params:<10}")


def print_model_summary_nested(model):
    """Just experimenting with alternate model summaries than model.summary()"""

    for layer in model.layers:
        print(layer.name)
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                print(f"  {sub_layer.name}")


def slerp(point1, point2, t):
    """Interpolation along a great-circle path between two points.
    Coord origin is the center of the hypersphere the great-circles are around.
    Used in interpolate_between_points() below.
    """
    omega = tf.acos(tf.clip_by_value(tf.tensordot(point1, point2, axes=1), -1.0, 1.0))
    sin_omega = tf.sin(omega)

    t1 = tf.sin((1 - t) * omega) / sin_omega
    t2 = tf.sin(t * omega) / sin_omega

    return t1 * point1 + t2 * point2


def interpolate_between_points(gaussian_points, N, path="euclidean"):
    """
    Interpolates N points between two high-dimensional points using a specified path.

    Parameters:
    gaussian_points (numpy array): A 2xD numpy array where D is the dimension of the points.
    N (int): Number of points to interpolate.
    path (str): The path type for interpolation ('euclidean' or 'slerp').

    Returns:
    numpy array: An NxD numpy array of interpolated points.
    """
    point1 = tf.constant(gaussian_points[0], dtype=tf.float32)
    point2 = tf.constant(gaussian_points[1], dtype=tf.float32)

    t_values = tf.linspace(0.0, 1.0, N)

    if path == "euclidean":
        interpolated_points = [(1 - t) * point1 + t * point2 for t in t_values]
    elif path == "slerp":
        interpolated_points = [slerp(point1, point2, t) for t in t_values]
    else:
        raise ValueError("Invalid path argument. Use 'euclidean' or 'slerp'.")

    return np.array(interpolated_points)
