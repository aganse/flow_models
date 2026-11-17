"""
Tests of the numpy-only forward models and priors used by
train_flowmodels5.py - train_flowmodels7.py.  No TensorFlow needed.
"""

import unittest
import numpy as np

import forward_models as fm


class TestGravprob(unittest.TestCase):

    def test_shapes(self):
        self.assertEqual(fm.GRAV_X.size, 51)
        self.assertEqual(fm.gravprob(np.full(51, 500.0)).shape, (1, 12))
        self.assertEqual(fm.gravprob(np.full((7, 51), 500.0)).shape, (7, 12))

    def test_matches_direct_quadrature(self):
        """Compare with fine-grid quadrature of Parker eqn 5.02(1)."""
        h = fm.GlacGravProblem.true_depths()
        d = fm.gravprob(h)[0]
        xf = np.linspace(0.0, fm.GRAV_A, 400001)
        hf = np.interp(xf, fm.GRAV_X, h)
        for j in (0, 5, 11):
            xs = fm.GRAV_XSTN[j]
            f = np.log(((xs - xf) ** 2 + hf ** 2) / np.maximum((xs - xf) ** 2, 1e-30))
            integral = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(xf))
            direct = integral * fm.GRAV_G * fm.GRAV_DRHO * 1e5
            self.assertAlmostEqual(d[j], direct, delta=0.1)

    def test_deeper_glacier_gives_more_negative_anomaly(self):
        shallow = fm.gravprob(np.full(51, 300.0))
        deep = fm.gravprob(np.full(51, 900.0))
        self.assertTrue(np.all(deep < shallow))


class TestGlacGravProblem(unittest.TestCase):

    def test_dimensions_for_each_configuration(self):
        expected = {
            ("smooth_gaussian", False): 51,
            ("smooth_gaussian", True): 63,
            ("smooth_gaussian_hierarchical", False): 52,
            ("smooth_gaussian_hierarchical", True): 64,
            ("uniform", False): 51,
            ("uniform", True): 63,
        }
        for (prior_type, noise), x_dim in expected.items():
            p = fm.GlacGravProblem(
                prior_type=prior_type, noise_sigma=0.5, noise_as_input=noise,
                n_stats=2000,
            )
            self.assertEqual(p.x_dim, x_dim)
            xs, ys = p.sample_training_batch(64)
            self.assertEqual(xs.shape, (64, x_dim))
            self.assertEqual(ys.shape, (64, 12))
            self.assertEqual(xs.dtype, np.float32)

    def test_prior_respects_bounds(self):
        p = fm.GlacGravProblem(prior_bounds=(0.0, 2000.0), n_stats=2000)
        h = p.depths(p.sample_inputs(5000))
        self.assertGreaterEqual(h.min(), 0.0)
        self.assertLessEqual(h.max(), 2000.0)
        u = fm.GlacGravProblem(
            prior_type="uniform", uniform_bounds=(100.0, 900.0), n_stats=2000
        )
        h = u.depths(u.sample_inputs(5000))
        self.assertGreaterEqual(h.min(), 100.0)
        self.assertLessEqual(h.max(), 900.0)

    def test_hierarchical_lambda_range(self):
        p = fm.GlacGravProblem(
            prior_type="smooth_gaussian_hierarchical",
            prior_scale=[0.01, 0.04], n_stats=2000,
        )
        log_lam = p.log10_lambda(p.sample_inputs(2000))
        self.assertGreaterEqual(log_lam.min(), np.log10(0.01) - 1e-9)
        self.assertLessEqual(log_lam.max(), np.log10(0.04) + 1e-9)

    def test_noise_enters_data(self):
        p = fm.GlacGravProblem(noise_sigma=0.5, noise_as_input=True, n_stats=2000)
        x = p.sample_inputs(100)
        diff = p.forward(x) - p.forward_noise_free(x)
        np.testing.assert_allclose(diff, p.noise(x))

    def test_standardization_roundtrip(self):
        p = fm.GlacGravProblem(n_stats=2000)
        x = p.sample_inputs(10)
        np.testing.assert_allclose(p.unstandardize_x(p.standardize_x(x)), x)
        y = p.forward(x)
        np.testing.assert_allclose(p.unstandardize_y(p.standardize_y(y)), y)

    def test_observed_data(self):
        p = fm.GlacGravProblem(n_stats=2000)
        self.assertEqual(p.observed_data("synthetic").shape, (12,))
        np.testing.assert_allclose(
            p.observed_data("parker"), fm.GRAV_DMEAS_PARKER
        )


class TestInvKinProblem(unittest.TestCase):

    def test_forward_matches_arm_end_point(self):
        p = fm.InvKinProblem(n_stats=2000)
        x = p.sample_inputs(50)
        y = p.forward(x)
        end = p.arm_points(x)[:, -1, :]  # (horizontal, vertical)
        np.testing.assert_allclose(y[:, 0], end[:, 1])
        np.testing.assert_allclose(y[:, 1], end[:, 0])

    def test_zero_angles_give_straight_arm(self):
        p = fm.InvKinProblem(n_stats=2000)
        y = p.forward(np.array([[0.3, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(y, [[0.3, 2.0]])

    def test_abc_posterior_hits_target(self):
        p = fm.InvKinProblem(n_stats=2000)
        target = np.array([0.0, 1.5])
        x, tol = p.abc_posterior(target, n_keep=200, n_prior=200000)
        self.assertEqual(x.shape, (200, 4))
        err = np.linalg.norm(p.forward(x) - target, axis=1)
        self.assertLessEqual(err.max(), tol + 1e-12)
        self.assertLess(tol, 0.2)


if __name__ == "__main__":
    unittest.main()
