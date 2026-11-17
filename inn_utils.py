"""Plotting and evaluation helpers shared by the INN-mode example scripts
(train_flowmodels3.py - train_flowmodels7.py).

Everything here works on numpy arrays (no TensorFlow), so the scripts do the
model calls and hand the results to these functions.
"""

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

def collect_pairs(generator, num_samples):
    """Accumulate exactly num_samples (x, y) rows from a generator that yields
    (x_batch, y_batch) pairs.  Returns two numpy arrays."""
    xs, ys = [], []
    total = 0
    while total < num_samples:
        x, y = next(generator)[:2]
        take = min(num_samples - total, x.shape[0])
        xs.append(np.asarray(x[:take], dtype=np.float32))
        ys.append(np.asarray(y[:take], dtype=np.float32))
        total += take
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def calibration_curve(x_true, x_post, num_q=50):
    """Posterior calibration as in Ardizzone et al 2019 Sec 4.2.

    x_true: (n_cases, n_params) true parameter values of test cases.
    x_post: (n_cases, n_samples, n_params) posterior samples for each case.
    For each confidence level q, q_inliers is the fraction of true values
    falling inside the central q-interval of their posterior samples; a
    well-calibrated posterior has q_inliers == q.

    Returns (q, q_inliers of shape (num_q, n_params), median abs calibration
    error per parameter).
    """
    x_true = np.asarray(x_true)
    x_post = np.asarray(x_post)
    rank = np.mean(x_post < x_true[:, None, :], axis=1)  # (n_cases, n_params)
    central = np.abs(2.0 * rank - 1.0)
    q = np.linspace(0.0, 1.0, num_q + 2)[1:-1]
    q_inliers = np.stack([np.mean(central <= qi, axis=0) for qi in q], axis=0)
    cal_err = np.median(np.abs(q_inliers - q[:, None]), axis=0)
    return q, q_inliers, cal_err


def prior_predictive_percentile(y_prior, y_obs):
    """Where does an observed data vector sit within the data predicted by
    the prior?  Returns the percentile (0-100) of its Mahalanobis distance
    among the prior-predicted data vectors y_prior (n, d).  Values near 100
    mean the observation lies outside what the network was trained on, so the
    posterior for it is an extrapolation and should not be trusted."""
    y_prior = np.asarray(y_prior, dtype=np.float64)
    mu = y_prior.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(y_prior, rowvar=False))
    dev = y_prior - mu
    d_prior = np.einsum("ij,jk,ik->i", dev, cov_inv, dev)
    dev_obs = np.asarray(y_obs, dtype=np.float64) - mu
    d_obs = float(dev_obs @ cov_inv @ dev_obs)
    return float(100.0 * np.mean(d_prior <= d_obs))


def mmd_numpy(a, b, bandwidths=(0.2, 1.5, 3.0), max_n=2000):
    """MMD between two sample sets with inverse multiquadratic kernels (same
    definition as FlowModel._mmd)."""
    a = np.asarray(a, dtype=np.float64)[:max_n]
    b = np.asarray(b, dtype=np.float64)[:max_n]

    def sq(p, q):
        d = (
            np.sum(p * p, axis=1)[:, None]
            + np.sum(q * q, axis=1)[None, :]
            - 2.0 * p @ q.T
        )
        return np.maximum(d, 0.0)

    daa, dbb, dab = sq(a, a), sq(b, b), sq(a, b)
    out = 0.0
    for h in bandwidths:
        h2 = h * h
        out += (
            np.mean(h2 / (h2 + daa))
            + np.mean(h2 / (h2 + dbb))
            - 2.0 * np.mean(h2 / (h2 + dab))
        )
    return float(out)


def latent_stats(z, prefix="latent"):
    """Summary of how closely latent samples z match N(0, I): dict of metrics
    named like the ones train_flowmodels2.py logs."""
    z = np.asarray(z)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    return {
        f"{prefix}_mean_of_means": float(mean.mean()),
        f"{prefix}_max_abs_mean": float(np.abs(mean).max()),
        f"{prefix}_mean_std": float(std.mean()),
        f"{prefix}_min_std": float(std.min()),
        f"{prefix}_max_std": float(std.max()),
    }


def confusion_matrix(true_idx, pred_idx, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(true_idx), np.asarray(pred_idx)):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# Generic plots
# ---------------------------------------------------------------------------

def _square_limits(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    m = max(
        abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
        abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]),
    )
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)


def plot_labeled_pts_2d(
    pts,
    label_idx,
    label_colors,
    plotfile,
    side="data",
    title=None,
    label_names=None,
    unit_circles=False,
    marker_pts=None,
    marker_label="class means",
    axis_labels=("x", "y"),
):
    """Scatterplot of 2D points colored by integer label (the colored analog
    of utils.plot_pts_2d).  unit_circles=True draws the circles containing
    50% and 90% of the mass of a standard 2D Gaussian.  marker_pts: optional
    (k, 2) points drawn as black crosses (e.g. class target positions)."""
    pts = np.asarray(pts)
    label_idx = np.asarray(label_idx)
    label_names = label_names or label_colors
    fig, ax = plt.subplots()
    for k, color in enumerate(label_colors):
        sel = label_idx == k
        if np.any(sel):
            ax.scatter(
                pts[sel, 0], pts[sel, 1], color=color, alpha=0.5, s=8,
                label=str(label_names[k]),
            )
    if unit_circles:
        th = np.linspace(0, 2 * np.pi, 200)
        for frac in (0.5, 0.9):
            r = np.sqrt(-2.0 * np.log(1.0 - frac))
            ax.plot(r * np.cos(th), r * np.sin(th), "k--", linewidth=1)
    if marker_pts is not None:
        marker_pts = np.asarray(marker_pts)
        ax.scatter(
            marker_pts[:, 0], marker_pts[:, 1], color="k", marker="x", s=80,
            linewidths=2, zorder=5, label=marker_label,
        )
    _square_limits(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title or f"2D points in {side} space")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def plot_label_space(y_pred, label_idx, label_colors, plotfile, label_names=None):
    """One panel per true label: mean (bar) and spread (error bar) of the
    predicted label vectors y for points of that label."""
    y_pred = np.asarray(y_pred)
    label_idx = np.asarray(label_idx)
    num = len(label_colors)
    label_names = label_names or label_colors
    fig, axes = plt.subplots(1, num, figsize=(3 * num, 3), sharey=True)
    axes = np.atleast_1d(axes)
    for k in range(num):
        ax = axes[k]
        sel = label_idx == k
        if np.any(sel):
            ax.bar(
                np.arange(y_pred.shape[1]),
                y_pred[sel].mean(axis=0),
                yerr=y_pred[sel].std(axis=0),
                color=label_colors[: y_pred.shape[1]],
                capsize=3,
            )
        ax.axhline(0.0, color="k", linewidth=0.5)
        ax.axhline(1.0, color="k", linewidth=0.5, linestyle=":")
        ax.set_xticks(np.arange(y_pred.shape[1]))
        ax.set_xticklabels(label_names[: y_pred.shape[1]], rotation=45)
        ax.set_title(f"true label: {label_names[k]}")
    axes[0].set_ylabel("predicted label vector y")
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def plot_latent_layout(
    x_grids, mode_centers, extent, plotfile, titles=None, closeness_scale=0.3
):
    """Layout of the latent space for fixed labels, as in Fig 8 of Ardizzone
    et al 2019: each latent position z on a grid is colored by which mixture
    mode its x = f^-1(y*, z) lands closest to (hue) and how close (brightness).

    x_grids: list of arrays (n_side, n_side, 2), one per fixed label y*, of the
        data-space points generated from a square grid of z values.
    mode_centers: (n_modes, 2) mixture mode centers in data space.
    extent: half-width of the z grid (grid spans [-extent, extent]^2).
    closeness_scale: distance from a mode at which brightness has dropped to
        about one third.
    Note the hues encode position x (nearest mode), not the label colors.
    """
    from matplotlib.colors import hsv_to_rgb

    mode_centers = np.asarray(mode_centers)
    n_modes = mode_centers.shape[0]
    num = len(x_grids)
    fig, axes = plt.subplots(1, num, figsize=(3.6 * num, 3.8))
    axes = np.atleast_1d(axes)
    th = np.linspace(0, 2 * np.pi, 200)
    for k, xg in enumerate(x_grids):
        xg = np.asarray(xg)
        d = np.linalg.norm(
            xg[:, :, None, :] - mode_centers[None, None, :, :], axis=-1
        )
        nearest = np.argmin(d, axis=-1)
        dist = np.min(d, axis=-1)
        hsv = np.stack(
            [
                nearest / n_modes,
                np.ones_like(dist),
                1.0 / (1.0 + 2.0 * (dist / closeness_scale) ** 2),
            ],
            axis=-1,
        )
        ax = axes[k]
        ax.imshow(
            hsv_to_rgb(hsv), origin="lower",
            extent=[-extent, extent, -extent, extent],
        )
        for frac in (0.5, 0.9):
            r = np.sqrt(-2.0 * np.log(1.0 - frac))
            ax.plot(r * np.cos(th), r * np.sin(th), "w-", linewidth=1)
        ax.set_xlabel("z1")
        if k == 0:
            ax.set_ylabel("z2")
        if titles:
            ax.set_title(titles[k])
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, plotfile, title="Confusion matrix"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2, 1.2 * len(class_names) + 1.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def plot_marginals(sample_sets, plotfile, names=None, truth=None, bins=40, ncols=4):
    """Histograms of each component for one or more sample sets.

    sample_sets: dict {legend label: (n, d) array}.
    truth: optional (d,) array of true values, drawn as vertical lines.
    """
    first = np.asarray(next(iter(sample_sets.values())))
    d = first.shape[1]
    names = names or [f"x{i + 1}" for i in range(d)]
    ncols = min(ncols, d)
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(d):
        ax = axes[i]
        lo = min(np.asarray(s)[:, i].min() for s in sample_sets.values())
        hi = max(np.asarray(s)[:, i].max() for s in sample_sets.values())
        edges = np.linspace(lo, hi if hi > lo else lo + 1.0, bins + 1)
        for k, (label, s) in enumerate(sample_sets.items()):
            ax.hist(
                np.asarray(s)[:, i], bins=edges, density=True, alpha=0.5,
                color=f"C{k}", label=label,
            )
        if truth is not None:
            ax.axvline(truth[i], color="k", linewidth=1.5)
        ax.set_title(names[i])
        ax.set_yticks([])
    for ax in axes[d:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(q, q_inliers, plotfile, names=None, title=None):
    """Calibration error q_inliers - q versus confidence q, one line per
    parameter (or their median if there are many)."""
    q_inliers = np.asarray(q_inliers)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    err = q_inliers - q[:, None]
    if err.shape[1] <= 8:
        for j in range(err.shape[1]):
            ax.plot(q, err[:, j], label=names[j] if names else f"x{j + 1}")
        ax.legend(fontsize=8)
    else:
        ax.fill_between(
            q, np.percentile(err, 5, axis=1), np.percentile(err, 95, axis=1),
            alpha=0.3, label="5-95% over parameters",
        )
        ax.plot(q, np.median(err, axis=1), label="median over parameters")
        ax.legend(fontsize=8)
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("confidence q")
    ax.set_ylabel("calibration error  q_inliers - q")
    ax.set_title(title or "Posterior calibration")
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def save_image_grid(image_rows, plotfile, row_labels=None, title=None):
    """Save a grid of images.  image_rows: list of arrays (n, H, W, C) with
    values in [0, 1] (clipped), one array per grid row."""
    nrows = len(image_rows)
    ncols = max(np.asarray(r).shape[0] for r in image_rows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.6 * ncols + 1.0, 1.6 * nrows + 0.4),
        squeeze=False,
    )
    for i, row in enumerate(image_rows):
        row = np.clip(np.asarray(row, dtype=np.float32), 0.0, 1.0)
        for j in range(ncols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j < row.shape[0]:
                img = row[j]
                if img.shape[-1] == 1:
                    ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img)
        if row_labels:
            axes[i, 0].set_ylabel(row_labels[i], fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(plotfile, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inverse kinematics plots
# ---------------------------------------------------------------------------

def plot_arm_posteriors(arm_point_sets, y_target, plotfile, max_arms=1000):
    """Side-by-side arm-configuration plots as in Fig 3 of Ardizzone et al
    2019.

    arm_point_sets: dict {panel title: (n, 4, 2) array of joint positions as
        (horizontal, vertical), from InvKinProblem.arm_points}.
    y_target: (vertical, horizontal) of the target end point, or None.
    """
    n_panels = len(arm_point_sets)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4.5), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (title, pts) in zip(axes, arm_point_sets.items()):
        pts = np.asarray(pts)[:max_arms]
        for arm in pts:
            ax.plot(arm[:, 0], arm[:, 1], color="C0", alpha=0.03, linewidth=1)
        ax.scatter(pts[:, -1, 0], pts[:, -1, 1], color="C0", s=2, alpha=0.2)
        if y_target is not None:
            ax.plot(
                [y_target[1]], [y_target[0]], marker="*", color="orange",
                markersize=14, markeredgecolor="k", linestyle="none",
                label="target y*",
            )
        ax.axvline(0.0, color="gray", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("horizontal")
    axes[0].set_ylabel("vertical")
    if y_target is not None:
        axes[0].legend(loc="lower left", fontsize=8)
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Glacier gravimetry plots
# ---------------------------------------------------------------------------

def plot_depth_posterior(
    x_nodes,
    post_depths,
    plotfile,
    stations=None,
    true_depths=None,
    prior_depths=None,
    extra_sets=None,
    title=None,
    num_draws=30,
    max_depth=None,
    band_label="posterior",
):
    """Posterior over glacier depth profiles: 5-95% and 25-75% bands, mean,
    a few individual samples, and optionally the true profile and the prior
    5-95% band.  Depth axis is positive downward as in gravinvParker.m.

    extra_sets: optional dict {label: (n, n_nodes) depths} whose means are
        overlaid as additional lines (e.g. samples at the MAP of lambda).
    """
    xk = np.asarray(x_nodes) / 1e3
    post_depths = np.asarray(post_depths)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if prior_depths is not None:
        lo, hi = np.percentile(prior_depths, [5, 95], axis=0)
        ax.plot(xk, lo, color="gray", linestyle=":", linewidth=1)
        ax.plot(xk, hi, color="gray", linestyle=":", linewidth=1,
                label="prior 5-95%")
    lo, hi = np.percentile(post_depths, [5, 95], axis=0)
    ax.fill_between(xk, lo, hi, color="C0", alpha=0.2, label=f"{band_label} 5-95%")
    lo, hi = np.percentile(post_depths, [25, 75], axis=0)
    ax.fill_between(xk, lo, hi, color="C0", alpha=0.35, label=f"{band_label} 25-75%")
    for h in post_depths[:num_draws]:
        ax.plot(xk, h, color="C0", alpha=0.25, linewidth=0.7)
    ax.plot(xk, post_depths.mean(axis=0), color="C0", linewidth=2.5,
            label=f"{band_label} mean")
    if extra_sets:
        for k, (label, h) in enumerate(extra_sets.items()):
            ax.plot(xk, np.asarray(h).mean(axis=0), color=f"C{k + 1}",
                    linewidth=2, linestyle="--", label=label)
    if true_depths is not None:
        ax.plot(xk, true_depths, color="k", linewidth=2.5, label="true model")
    if stations is not None:
        ax.plot(np.asarray(stations) / 1e3, np.zeros(len(stations)), "k^",
                markerfacecolor="k", label="meas stns")
    ax.set_xlabel("cross range x (km)")
    ax.set_ylabel("depth (m)")
    ax.set_xlim(xk.min(), xk.max())
    if max_depth is None:
        max_depth = 1.05 * min(
            max(1400.0, float(np.percentile(post_depths, 99))), 3000.0
        )
    ax.set_ylim(max_depth, -0.02 * max_depth)  # positive down
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    if title:
        ax.set_title(title)
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def plot_data_fit(stations, d_obs, d_pred, plotfile, noise_sigma=0.0, title=None):
    """Observed gravity anomalies versus the spread of data re-simulated from
    posterior model samples (d_pred: (n, n_stations))."""
    xs = np.asarray(stations) / 1e3
    d_pred = np.asarray(d_pred)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    ax = axes[0]
    lo, hi = np.percentile(d_pred, [5, 95], axis=0)
    ax.fill_between(xs, lo, hi, color="C0", alpha=0.3,
                    label="re-simulated 5-95%")
    ax.plot(xs, d_pred.mean(axis=0), color="C0", label="re-simulated mean")
    if noise_sigma > 0:
        ax.errorbar(xs, d_obs, yerr=noise_sigma, fmt="k.", capsize=3,
                    label="observed (+/- noise std)")
    else:
        ax.plot(xs, d_obs, "k.", markersize=9, label="observed")
    ax.set_ylabel("gravity anomaly (mGal)")
    ax.legend(fontsize=8)
    ax.grid(True)
    if title:
        ax.set_title(title)
    ax = axes[1]
    resid = d_pred - np.asarray(d_obs)[None, :]
    lo, hi = np.percentile(resid, [5, 95], axis=0)
    ax.fill_between(xs, lo, hi, color="C0", alpha=0.3)
    ax.plot(xs, resid.mean(axis=0), color="C0")
    if noise_sigma > 0:
        ax.axhline(noise_sigma, color="k", linestyle=":", linewidth=1)
        ax.axhline(-noise_sigma, color="k", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("cross range x (km)")
    ax.set_ylabel("residual (mGal)")
    ax.grid(True)
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def plot_depth_std(x_nodes, std_sets, plotfile, title=None):
    """Depth uncertainty (std per node) for several sample sets, e.g. prior
    versus posterior.  std_sets: dict {label: (n, n_nodes) depths}."""
    xk = np.asarray(x_nodes) / 1e3
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, h in std_sets.items():
        ax.plot(xk, np.asarray(h).std(axis=0), linewidth=2, label=label)
    ax.set_xlabel("cross range x (km)")
    ax.set_ylabel("depth std (m)")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def map_window(values, frac=0.2, bins=30):
    """Locate the mode (MAP) of 1D samples by histogram and return
    (map_value, boolean mask of the fraction `frac` of samples closest to
    it)."""
    values = np.asarray(values)
    counts, edges = np.histogram(values, bins=bins)
    k = int(np.argmax(counts))
    map_value = 0.5 * (edges[k] + edges[k + 1])
    dist = np.abs(values - map_value)
    cutoff = np.quantile(dist, frac)
    return float(map_value), dist <= cutoff


def plot_lambda_posterior(prior_log_lam, post_log_lam, plotfile, map_value=None, title=None):
    """Prior versus posterior histogram of log10(lambda), the prior-strength
    hyperparameter of the hierarchical smooth-Gaussian prior."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    edges = np.linspace(
        min(np.min(prior_log_lam), np.min(post_log_lam)),
        max(np.max(prior_log_lam), np.max(post_log_lam)),
        41,
    )
    ax.hist(prior_log_lam, bins=edges, density=True, alpha=0.4,
            color="gray", label="prior")
    ax.hist(post_log_lam, bins=edges, density=True, alpha=0.6,
            color="C0", label="posterior")
    if map_value is not None:
        ax.axvline(map_value, color="k", linestyle="--", label="posterior MAP")
    ax.set_xlabel("log10(lambda)")
    ax.set_yticks([])
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)
