"""Forward models, priors, and standardization for the INN example problems
(train_flowmodels5.py: inverse kinematics; train_flowmodels6.py and
train_flowmodels7.py: glacier-depth gravimetry).

Pure numpy (no TensorFlow) so the physics can be tested on its own.

Each problem object provides:
    x_dim, y_dim            sizes of the (unpadded) input and data vectors
    sample_inputs(n)        draw n raw input vectors x from the prior
    forward(x)              raw data vectors y for raw inputs x
    standardize_x/_y and unstandardize_x/_y
    sample_training_batch(n)   (x_std, y_std) pair for training

The INN is trained on standardized values (zero mean, unit std per component,
using fixed statistics estimated once from the prior) so that the MMD kernel
widths and loss weights have comparable meaning across problems.
"""

import numpy as np


class _Problem:
    """Shared standardization machinery."""

    x_dim = None
    y_dim = None

    def _init_stats(self, n_stats=20000, stats_seed=0):
        rng = np.random.default_rng(stats_seed)
        x = self.sample_inputs(n_stats, rng=rng)
        y = self.forward(x)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-12
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-12

    def standardize_x(self, x):
        return (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std

    def unstandardize_x(self, xs):
        return np.asarray(xs, dtype=np.float64) * self.x_std + self.x_mean

    def standardize_y(self, y):
        return (np.asarray(y, dtype=np.float64) - self.y_mean) / self.y_std

    def unstandardize_y(self, ys):
        return np.asarray(ys, dtype=np.float64) * self.y_std + self.y_mean

    def sample_training_batch(self, n, rng=None):
        x = self.sample_inputs(n, rng=rng)
        y = self.forward(x)
        return (
            self.standardize_x(x).astype(np.float32),
            self.standardize_y(y).astype(np.float32),
        )


# ---------------------------------------------------------------------------
# Inverse kinematics (Ardizzone et al 2019, Sec 4.1 and appendix Sec 3)
# ---------------------------------------------------------------------------

class InvKinProblem(_Problem):
    """Articulated arm in 2D: x = (vertical offset of base, 3 joint angles),
    y = (vertical, horizontal) position of the arm's end point.

    Follows eqns (13)-(14) of the appendix of Ardizzone et al 2019:
        y1 = x1 + l1 sin(x2) + l2 sin(x3 - x2) + l3 sin(x4 - x2 - x3)
        y2 =      l1 cos(x2) + l2 cos(x3 - x2) + l3 cos(x4 - x2 - x3)
    with priors x_i ~ N(0, sigma_i^2).
    """

    name = "invkin"
    x_dim = 4
    y_dim = 2

    def __init__(
        self,
        sigmas=(0.25, 0.5, 0.5, 0.5),
        lengths=(0.5, 0.5, 1.0),
        n_stats=20000,
        stats_seed=0,
    ):
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self._init_stats(n_stats, stats_seed)

    def sample_inputs(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return rng.normal(size=(n, 4)) * self.sigmas

    def _segment_angles(self, x):
        x = np.atleast_2d(x)
        return np.stack(
            [x[:, 1], x[:, 2] - x[:, 1], x[:, 3] - x[:, 1] - x[:, 2]], axis=1
        )

    def forward(self, x):
        x = np.atleast_2d(x)
        ang = self._segment_angles(x)
        y1 = x[:, 0] + np.sum(self.lengths * np.sin(ang), axis=1)
        y2 = np.sum(self.lengths * np.cos(ang), axis=1)
        return np.stack([y1, y2], axis=1)

    def arm_points(self, x):
        """Joint positions for plotting: array (n, 4, 2) of (horizontal,
        vertical) coordinates for base, joint 1, joint 2, end point."""
        x = np.atleast_2d(x)
        ang = self._segment_angles(x)
        dh = self.lengths * np.cos(ang)
        dv = self.lengths * np.sin(ang)
        horiz = np.concatenate(
            [np.zeros((x.shape[0], 1)), np.cumsum(dh, axis=1)], axis=1
        )
        vert = x[:, :1] + np.concatenate(
            [np.zeros((x.shape[0], 1)), np.cumsum(dv, axis=1)], axis=1
        )
        return np.stack([horiz, vert], axis=2)

    def abc_posterior(self, y_target, n_keep=2000, n_prior=2000000, rng=None):
        """Rejection ABC reference posterior: draw n_prior prior samples and
        keep the n_keep whose forward-modelled end points are closest to
        y_target.  Returns (x_samples, tolerance) where tolerance is the
        largest accepted distance."""
        rng = rng or np.random.default_rng(1)
        y_target = np.asarray(y_target, dtype=np.float64).reshape(1, 2)
        chunk = 500000
        keep_x = []
        keep_d = []
        done = 0
        while done < n_prior:
            m = min(chunk, n_prior - done)
            x = self.sample_inputs(m, rng=rng)
            d = np.linalg.norm(self.forward(x) - y_target, axis=1)
            idx = np.argpartition(d, min(n_keep, m - 1))[:n_keep]
            keep_x.append(x[idx])
            keep_d.append(d[idx])
            done += m
        keep_x = np.concatenate(keep_x, axis=0)
        keep_d = np.concatenate(keep_d, axis=0)
        idx = np.argsort(keep_d)[:n_keep]
        return keep_x[idx], float(keep_d[idx].max())


# ---------------------------------------------------------------------------
# Glacier-depth gravimetry (Parker 1994 ch.5; port of gravprob.m)
# ---------------------------------------------------------------------------

GRAV_G = 6.675e-11  # m^3/kg/s^2
GRAV_DRHO = -1700.0  # kg/m^3  (ice 1000 minus rock 2700)
GRAV_A = 3420.0  # m, horizontal extent of valley
GRAV_XINCR = 68.4  # m, node spacing
GRAV_X = np.arange(0.0, GRAV_A + 0.5 * GRAV_XINCR, GRAV_XINCR)  # 51 nodes
GRAV_XSTN = np.array(
    [535, 749, 963, 1177, 1391, 1605, 1819, 2033, 2247, 2461, 2675, 2889],
    dtype=np.float64,
)  # station locations (m)
# Parker's corrected gravity anomaly measurements (mGal) at GRAV_XSTN:
GRAV_DMEAS_PARKER = np.array(
    [-15.0, -24.0, -31.2, -36.8, -40.8, -42.7,
     -42.4, -40.9, -37.3, -31.5, -21.8, -12.8]
)


def gravprob(h):
    """Nonlinear gravity forward problem, vectorized port of gravprob.m.

    h: array (n, 51) or (51,) of glacier depths (m, positive down) at GRAV_X.
    Returns gravity anomalies in mGal at the 12 stations, shape (n, 12).

    Parker eqn 5.05(26): the log singularity of the kernel is removed
    analytically, and the remaining smooth part is integrated by trapezoid:
      dg_j/(G drho) = int_0^a ln[(x-x_j)^2 + h(x)^2] dx
                      - 2[(a-x_j)(ln|x_j-a| - 1) + x_j ln(x_j) - x_j]
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    dx2 = (GRAV_X[None, :] - GRAV_XSTN[:, None]) ** 2  # (12, 51)
    y = np.log(dx2[None, :, :] + h[:, None, :] ** 2)  # (n, 12, 51)
    integral = GRAV_XINCR * (
        y.sum(axis=2) - 0.5 * (y[:, :, 0] + y[:, :, -1])
    )
    xs = GRAV_XSTN
    analytic = 2.0 * (
        (GRAV_A - xs) * (np.log(np.abs(xs - GRAV_A)) - 1.0)
        + xs * np.log(xs)
        - xs
    )
    dg = GRAV_G * GRAV_DRHO * (integral - analytic[None, :])
    return dg * 1e5  # m/s^2 -> mGal


def second_diff_operator(n):
    """(n-2) x n second-difference matrix with rows [1, -2, 1]."""
    d2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        d2[i, i : i + 3] = [1.0, -2.0, 1.0]
    return d2


def _reflect(h, lb, ub):
    """Fold values back inside [lb, ub] (keeps profiles continuous and avoids
    the point masses at the bounds that clipping would create)."""
    h = lb + np.abs(h - lb)
    return ub - np.abs(ub - h)


class GlacGravProblem(_Problem):
    """Bayesian form of the glacier-depth gravimetry problem.

    Model: depth profile h(x) at the 51 nodes of GRAV_X (m, positive down).
    Data:  gravity anomaly (mGal) at the 12 stations of GRAV_XSTN.

    prior_type:
      "smooth_gaussian": h ~ N(prior_mean, C) folded into prior_bounds, with
          C^-1 = prior_scale^2 * L^T L,  L = [D2; prior_damping * I],
          i.e. the Occam regularization operator of gravinvParker.m used as
          the prior precision.  prior_scale plays the role of Occam's lambda
          and is a fixed scalar here.
      "smooth_gaussian_hierarchical": as above, but lambda is itself random,
          log10(lambda) ~ Uniform over log10 of prior_scale = [low, high], and
          log10(lambda) is appended to the model vector (so it is estimated
          along with the depths).
      "uniform": h_i ~ Uniform(uniform_bounds) independently per node
          (as in Zhang & Curtis 2021).

    noise_sigma: std (mGal) of Gaussian data noise.  When noise_as_input is
      True (use case 7) the noise vector is part of the network input,
      x = [model, noise], per Zhang & Curtis 2021 eqns 10-13, and
      forward(x) = gravprob(h) + noise.  When False (use case 6) there is no
      noise anywhere.
    """

    name = "glacgrav"
    n_depth = GRAV_X.size
    y_dim = GRAV_XSTN.size

    def __init__(
        self,
        prior_type="smooth_gaussian",
        prior_scale=0.02,
        prior_mean=600.0,
        prior_damping=0.05,
        prior_bounds=(0.0, 10000.0),
        uniform_bounds=(0.0, 1500.0),
        noise_sigma=0.0,
        noise_as_input=False,
        n_stats=20000,
        stats_seed=0,
    ):
        valid = ("smooth_gaussian", "smooth_gaussian_hierarchical", "uniform")
        if prior_type not in valid:
            raise ValueError(f"prior_type must be one of {valid}")
        self.prior_type = prior_type
        self.prior_mean = float(prior_mean)
        self.prior_damping = float(prior_damping)
        self.prior_bounds = tuple(float(b) for b in prior_bounds)
        self.uniform_bounds = tuple(float(b) for b in uniform_bounds)
        self.noise_sigma = float(noise_sigma)
        self.has_noise = bool(noise_as_input) and self.noise_sigma > 0
        self.has_lambda = prior_type == "smooth_gaussian_hierarchical"

        if self.has_lambda:
            if np.isscalar(prior_scale):
                prior_scale = [prior_scale / 3.0, prior_scale * 3.0]
            self.log10_lambda_range = tuple(
                float(np.log10(v)) for v in prior_scale
            )
            self.prior_scale = None
        else:
            if not np.isscalar(prior_scale):
                # tolerate a [low, high] range by using its geometric mean
                prior_scale = float(np.sqrt(prior_scale[0] * prior_scale[1]))
            self.prior_scale = float(prior_scale)
            self.log10_lambda_range = None

        # Unit-lambda prior factor: cov = (L^T L)^-1 = A A^T
        n = self.n_depth
        d2 = second_diff_operator(n)
        q1 = d2.T @ d2 + self.prior_damping ** 2 * np.eye(n)
        chol = np.linalg.cholesky(q1)
        self._prior_factor = np.linalg.inv(chol).T
        self.unit_prior_std = np.sqrt(
            np.sum(self._prior_factor ** 2, axis=1)
        )  # per-node marginal std for lambda=1 (before folding at bounds)

        self.n_model = n + (1 if self.has_lambda else 0)
        self.x_dim = self.n_model + (self.y_dim if self.has_noise else 0)
        self._init_stats(n_stats, stats_seed)
        if self.has_noise:
            # noise components: exactly zero-mean with known std
            self.x_mean[self.n_model :] = 0.0
            self.x_std[self.n_model :] = self.noise_sigma

    # -- prior ----------------------------------------------------------
    def sample_model(self, n, rng=None):
        """Draw n model vectors: (n, 51) depths, plus a final log10(lambda)
        column for the hierarchical prior."""
        rng = rng or np.random.default_rng()
        nd = self.n_depth
        if self.prior_type == "uniform":
            lo, hi = self.uniform_bounds
            return rng.uniform(lo, hi, size=(n, nd))
        xi = rng.normal(size=(n, nd)) @ self._prior_factor.T
        if self.has_lambda:
            lo, hi = self.log10_lambda_range
            log_lam = rng.uniform(lo, hi, size=(n, 1))
            lam = 10.0 ** log_lam
        else:
            lam = self.prior_scale
        h = self.prior_mean + xi / lam
        h = _reflect(h, *self.prior_bounds)
        if self.has_lambda:
            return np.concatenate([h, log_lam], axis=1)
        return h

    def sample_inputs(self, n, rng=None):
        rng = rng or np.random.default_rng()
        m = self.sample_model(n, rng=rng)
        if self.has_noise:
            eps = rng.normal(size=(n, self.y_dim)) * self.noise_sigma
            return np.concatenate([m, eps], axis=1)
        return m

    # -- pieces of the input vector --------------------------------------
    def depths(self, x):
        return np.atleast_2d(x)[:, : self.n_depth]

    def log10_lambda(self, x):
        if not self.has_lambda:
            return None
        return np.atleast_2d(x)[:, self.n_depth]

    def noise(self, x):
        if not self.has_noise:
            return None
        return np.atleast_2d(x)[:, self.n_model :]

    # -- forward ----------------------------------------------------------
    def forward(self, x):
        """Data predicted for input vectors x (including the noise part of x
        when noise is a network input)."""
        d = gravprob(self.depths(x))
        if self.has_noise:
            d = d + self.noise(x)
        return d

    def forward_noise_free(self, x):
        return gravprob(self.depths(x))

    # -- reference models / observed data ---------------------------------
    @staticmethod
    def true_depths():
        """Synthetic 'true' profile of gravinvParker.m (like Parker's)."""
        return 1000.0 * np.sin(np.pi * GRAV_X / GRAV_A) ** 2

    def observed_data(self, which="synthetic", seed=0):
        """Observed data vector d* (mGal).
        which="synthetic": gravprob(true_depths()) plus one noise realization
            of std noise_sigma (if noise_sigma > 0);
        which="parker": Parker's 12 measured anomalies."""
        if which == "parker":
            return GRAV_DMEAS_PARKER.copy()
        d = gravprob(self.true_depths())[0]
        if self.noise_sigma > 0:
            rng = np.random.default_rng(seed)
            d = d + rng.normal(size=d.shape) * self.noise_sigma
        return d
