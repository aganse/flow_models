import itertools
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pprint
from scipy.spatial import distance
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf


def _unwrap_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def imgs_to_gaussian_pts(
    model,
    image_generator,
    N,
    neigvals=100,
    p_outliers=10,
    chunk_size=None,
    pca_solver="auto",
    return_input_samples=False,
):
    """Map input images (from data generator) through the model to points in the
    Gaussian latent space.  Also computes latent space stats in reduced coords
    (via pca, since too high dimensionality for later sampling).  Also computes
    the #p_outliers most extreme outliers (re euclidean dist) and #p_outliers
    points closest to the mean.

    Images coming out of image_generator are (MxM) pixels.
    N = number of images to draw from image_generator to map.
    chunk_size: optional size of minibatches for mapping to latent space to avoid
      OOM; defaults to N (all at once).
    Make sure neigvals<<M^2.
    return_input_samples: when True, returns an additional numpy array containing
      the flattened input samples that produced the latent points.
    """

    first_batch = _unwrap_batch(next(image_generator))
    image_generator = itertools.chain([first_batch], image_generator)
    M = np.prod(first_batch.shape[1:])
    # Allow caller to override PCA dimensionality; keep legacy default when None.
    if neigvals is None:
        pca_n_components = None
    else:
        pca_n_components = min(M, N, neigvals)
    chunk_size = min(N, chunk_size or N)

    def image_chunks(data_generator, n, chunk):
        """Yield up to n images from data_generator in minibatches of size chunk."""
        collected = 0
        buffer = []
        while collected < n:
            img_batch = _unwrap_batch(next(data_generator))
            for img in img_batch:
                buffer.append(img)
                collected += 1
                if len(buffer) == chunk or collected == n:
                    yield np.array(buffer, dtype=np.float32)
                    buffer = []
                if collected == n:
                    break

    print("images.shape 1:", (N, *first_batch.shape[1:]))
    print("images.shape 2:", (N, int(M)))
    gaussian_chunks = []
    input_chunks = [] if return_input_samples else None
    for img_chunk in image_chunks(image_generator, N, chunk_size):
        flat_chunk = tf.reshape(
            tf.convert_to_tensor(img_chunk, dtype=tf.float32), (-1, M)
        )
        gaussian_chunk = model.call(flat_chunk)
        gaussian_chunks.append(gaussian_chunk.numpy())
        if return_input_samples:
            input_chunks.append(np.reshape(img_chunk, (img_chunk.shape[0], -1)))

    gaussian_points = np.concatenate(gaussian_chunks, axis=0)
    print("first several gaussian pts:", gaussian_points[:8])
    print("gpts.shape 3:", gaussian_points.shape)
    input_samples = (
        np.concatenate(input_chunks, axis=0) if return_input_samples else None
    )

    if N > 10:
        mean_full = np.mean(gaussian_points, axis=0)

        if pca_n_components not in (None, 0):
            pca = PCA(n_components=pca_n_components, svd_solver=pca_solver)
            reduced_data = pca.fit_transform(gaussian_points)
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
            pca = None
            cov_reduced = None
            top_outliers = None
            closest_to_mean = None
    else:
        mean_full = None
        cov_reduced = None
        pca = None
        top_outliers = None
        closest_to_mean = None

    if return_input_samples:
        return (
            gaussian_points,
            mean_full,
            cov_reduced,
            pca,
            top_outliers,
            closest_to_mean,
            input_samples,
        )
    return gaussian_points, mean_full, cov_reduced, pca, top_outliers, closest_to_mean


def plot_pts_2d(  # noqa: C901
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
    highlight_pts=None,
    highlight_label="highlighted pts",
    highlight_color="orange",
):
    """Scatterplot of various categories of points in the Gaussian latent space."""

    train_pts = np.asarray(train_pts)
    if sim_pts is not None:
        sim_pts = np.asarray(sim_pts)
    if other_pts is not None:
        other_pts = np.asarray(other_pts)
    if highlight_pts is not None:
        highlight_pts = np.asarray(highlight_pts)
    if mean is not None:
        mean = np.asarray(mean)

    print("plot_pts_2d train_pts.shape:", train_pts.shape)
    if train_pts.shape[1] > 2:
        print("applying PCA to reduce dimensions to 2 to plot...")
        pca = PCA(n_components=2)
        pca_inputs = [train_pts]
        if sim_pts is not None:
            pca_inputs.append(sim_pts)
        if other_pts is not None:
            pca_inputs.append(other_pts)
        if highlight_pts is not None:
            pca_inputs.append(highlight_pts)
        if mean is not None:
            pca_inputs.append(np.reshape(mean, (1, -1)))
        mix_pts = np.concatenate(pca_inputs, axis=0)
        pca.fit(mix_pts)
        train_pts = pca.transform(train_pts)
        if sim_pts is not None:
            sim_pts = pca.transform(sim_pts)
        if other_pts is not None:
            other_pts = pca.transform(other_pts)
        if highlight_pts is not None:
            highlight_pts = pca.transform(highlight_pts)
        if mean is not None:
            mean = pca.transform(np.reshape(mean, (1, -1)))[0]

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

    if highlight_pts is not None and highlight_pts.size > 0:
        ax.scatter(
            highlight_pts[:, 0],
            highlight_pts[:, 1],
            color=highlight_color,
            label=highlight_label,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )

    # autoset the plot limits to smallest square containing all points
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale()
    m = max(
        abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
        abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]),
    )
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)

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


def generate_multivariate_normal_samples(
    mean, reduced_cov, pca, num_samples, cov_scale=1.0
):
    """Used by generate_imgs_in_batches().  The high dimensionality requires
    generating samples in reduced space (via pca, hence reduced_cov), and then
    transforming back out to full dimension, and thus this function.
    """

    # Generate new samples in reduced space
    if pca is None or reduced_cov is None:
        raise ValueError("PCA-based sampling requested but pca or reduced_cov is None.")

    new_samples_reduced = np.random.multivariate_normal(
        # (make 1D mean into 2D, then rotate/reduce it, then put back to 1D)
        mean=np.squeeze(pca.transform([mean])),
        cov=reduced_cov * cov_scale,
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
    sampling_mode="pca",
    cov_scale=1.0,
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
    sampling_mode: "pca" (PCA-based sampling) or "direct" (N(0,I) latent sampling).
    cov_scale: multiplicative scaling on reduced_cov when sampling via PCA.
    """

    num_batches = (num_gen_images + batch_size - 1) // batch_size
    flat_dim = int(np.prod(model.image_shape))
    all_latent_samples = []

    for batch_idx in range(num_batches):
        # Determine the number of images to generate in this batch
        current_batch_size = min(batch_size, num_gen_images - batch_idx * batch_size)

        if regen_pts is None:
            if sampling_mode == "direct":
                # Sample from N(0,I) in latent space; decode via model.inverse().
                # Keeping latent samples explicit (not via TransformedDistribution.sample)
                # ensures the returned points are true latent coordinates for scatter plots.
                samples_tf = tf.random.normal(
                    shape=(current_batch_size, flat_dim), dtype=tf.float32
                )
            elif sampling_mode == "pca":
                samples_tf = generate_multivariate_normal_samples(
                    mean, reduced_cov, pca, current_batch_size, cov_scale=cov_scale
                )
            else:
                raise ValueError(f"Unknown sampling_mode '{sampling_mode}'")
        else:
            # Get next batch worth of points from supplied training_points
            regen_tf = tf.convert_to_tensor(regen_pts, dtype=tf.float32)
            samples_tf = regen_tf[
                (batch_idx * batch_size) : (batch_idx * batch_size + current_batch_size)
            ]

        all_latent_samples.append(np.reshape(np.array(samples_tf), (current_batch_size, -1)))

        for i in range(current_batch_size):
            generated_image = model.inverse(samples_tf[i : i + 1])
            generated_image = tf.reshape(generated_image, model.image_shape)

            # Save the generated image
            img = generated_image.numpy()
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
            img_idx = batch_idx * batch_size + i + 1
            if add_plot_num:
                img = add_text_to_image(
                    img, str(img_idx), font_size=20, color="orange", bold=True
                )
            plt.imsave(f"{filename}_{img_idx}.png", img)

        print(
            f"Generated and saved {batch_idx * batch_size + current_batch_size} images out of {num_gen_images}"
        )

    return np.concatenate(all_latent_samples, axis=0)


def generate_sim_pts(
    model,
    num_gen_images,
    mean=None,
    reduced_cov=None,
    pca=None,
    regen_pts=None,
    batch_size=10,
    sampling_mode="pca",
    cov_scale=1.0,
):
    """Sample latent points and map them back through the model.

    Returns:
        tuple(np.ndarray, np.ndarray): (simulated_data_points, latent_points)
    """

    num_batches = (num_gen_images + batch_size - 1) // batch_size

    data_batches = []
    latent_batches = []
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_gen_images - batch_idx * batch_size)

        if regen_pts is None:
            if sampling_mode == "direct":
                latent_dim = np.prod(model.image_shape)
                samples_tf = tf.random.normal(
                    shape=(current_batch_size, latent_dim), dtype=tf.float32
                )
                if cov_scale != 1.0:
                    samples_tf = samples_tf * cov_scale
            elif sampling_mode == "pca":
                samples_tf = generate_multivariate_normal_samples(
                    mean, reduced_cov, pca, current_batch_size, cov_scale=cov_scale
                )
            else:
                raise ValueError(f"Unknown sampling_mode '{sampling_mode}'")
        else:
            regen_tf = tf.convert_to_tensor(regen_pts, dtype=tf.float32)
            samples_tf = regen_tf[
                (batch_idx * batch_size) : (batch_idx * batch_size + current_batch_size)
            ]

        latent_batches.append(np.asarray(samples_tf, dtype=np.float32))

        generated_batch = model.inverse(samples_tf)
        generated_batch = tf.reshape(
            generated_batch,
            (current_batch_size, *model.image_shape),
        )
        data_batches.append(generated_batch.numpy())

    sim_data_pts = np.concatenate(data_batches, axis=0)
    sim_latent_pts = np.concatenate(latent_batches, axis=0)

    return sim_data_pts, sim_latent_pts


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

    print("Run params:")
    print(kwargs)

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
