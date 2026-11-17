"""
Tests of FlowModel's INN mode (y_dim > 0), i.e. x <-> [y, z] training on
(x, y) pairs as used by train_flowmodels3.py - train_flowmodels7.py.
All tests run on CPU with tiny synthetic tensors; no GPU, MLflow, or image
data needed.
"""

import time
import unittest
import numpy as np
import tensorflow as tf

from file_utils import get_data_generator
from flow_model import FlowModel

tf.get_logger().setLevel("ERROR")


def _one_train_step(model, x, y):
    ds = tf.data.Dataset.from_tensors((x, y))
    return model.fit(ds, steps_per_epoch=1, epochs=1, verbose=0)


class _Timed(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))


class TestInnModeMMDWithPadding(_Timed):
    """Ardizzone-style losses with padding on both sides - matches the
    train_flowmodels4.py case (2D points, 4 labels, 2 latents, width 16)."""

    @classmethod
    def setUpClass(cls):
        cls.model = FlowModel(
            image_shape=(16,),
            bijector="realnvp-based",
            realnvp_flow_steps=2,
            realnvp_hidden_layers=[8],
            x_dim=2,
            y_dim=4,
            z_dim=2,
            lambda_nll=0.0,
        )
        cls.model.compile(optimizer="adam")
        x = np.zeros((8, 16), dtype=np.float32)
        x[:, :2] = np.random.normal(size=(8, 2))
        cls.x = tf.constant(x)
        cls.y = tf.one_hot(np.arange(8) % 4, 4)
        _ = cls.model(cls.x)

    def test_dimension_bookkeeping(self):
        self.assertEqual(self.model.x_pad_dim, 14)
        self.assertEqual(self.model.out_pad_dim, 10)

    def test_train_step_runs_and_reports_terms(self):
        history = _one_train_step(self.model, self.x, self.y)
        for key in ("loss", "loss_y", "loss_z", "loss_x", "loss_pad",
                    "nll_per_dim", "bits_per_dim"):
            self.assertIn(key, history.history)
            self.assertTrue(np.isfinite(history.history[key][0]))

    def test_forward_yz_shapes(self):
        y, z, pad = self.model.forward_yz(self.x.numpy()[:, :2])
        self.assertEqual(y.shape, (8, 4))
        self.assertEqual(z.shape, (8, 2))
        self.assertEqual(pad.shape, (8, 10))

    def test_sample_posterior_shapes(self):
        x = self.model.sample_posterior([1.0, 0.0, 0.0, 0.0], 5)
        self.assertEqual(x.shape, (5, 2))
        x = self.model.sample_posterior(
            np.eye(4, dtype=np.float32), z=np.zeros((4, 2), dtype=np.float32)
        )
        self.assertEqual(x.shape, (4, 2))

    def test_forward_then_backward_recovers_input(self):
        u = self.model.call(self.x)
        x_back = self.model.inverse(u).numpy()
        np.testing.assert_allclose(x_back, self.x.numpy(), atol=1e-3)

    def test_mmd_is_zero_for_identical_samples(self):
        a = tf.random.normal([16, 3])
        self.assertAlmostEqual(float(self.model._mmd(a, a)), 0.0, places=5)
        b = a + 5.0
        self.assertGreater(float(self.model._mmd(a, b)), 0.1)

    def test_requires_xy_pairs(self):
        ds = tf.data.Dataset.from_tensors(self.x)
        with self.assertRaises(Exception):
            self.model.fit(ds, steps_per_epoch=1, epochs=1, verbose=0)

    def test_config_roundtrip(self):
        config = self.model.get_config()
        self.assertEqual(config["y_dim"], 4)
        clone = FlowModel.from_config(config)
        self.assertEqual(clone.z_dim, 2)
        self.assertEqual(clone.x_dim, 2)


class TestInnModeLikelihood(_Timed):
    """Likelihood-based INN training without padding - the form used by
    train_flowmodels6.py / train_flowmodels7.py."""

    @classmethod
    def setUpClass(cls):
        cls.model = FlowModel(
            image_shape=(8,),
            bijector="realnvp-based",
            realnvp_flow_steps=2,
            realnvp_hidden_layers=[8],
            y_dim=3,
            lambda_nll=1.0,
            lambda_z=1.0,
            lambda_x=1.0,
            y_sigma=0.1,
            grad_norm_thresh=10.0,
        )
        cls.model.compile(optimizer="adam")
        cls.x = tf.random.normal([8, 8])
        cls.y = tf.random.normal([8, 3])
        _ = cls.model(cls.x)

    def test_defaults(self):
        self.assertEqual(self.model.z_dim, 5)
        self.assertEqual(self.model.x_dim, 8)
        self.assertEqual(self.model.out_pad_dim, 0)

    def test_train_step_runs(self):
        history = _one_train_step(self.model, self.x, self.y)
        self.assertTrue(np.isfinite(history.history["loss"][0]))
        self.assertIn("preclip_grad_norm", history.history)

    def test_nll_matches_shifted_base_density(self):
        """With y_true = 0 and y_sigma = 1 the likelihood term must equal the
        plain flow's negative log-likelihood per dimension."""
        model = FlowModel(
            image_shape=(8,),
            bijector="realnvp-based",
            realnvp_flow_steps=2,
            realnvp_hidden_layers=[8],
            y_dim=3,
            lambda_nll=1.0,
            y_sigma=1.0,
        )
        _ = model(self.x)
        _, terms = model._inn_losses(self.x, tf.zeros([8, 3]))
        plain = -tf.reduce_mean(model.log_prob(self.x)) / 8.0
        self.assertAlmostEqual(
            float(terms["nll_per_dim"]), float(plain), places=3
        )

    def test_sample_posterior_adds_y_jitter_by_default(self):
        x = self.model.sample_posterior(np.zeros(3), 6)
        self.assertEqual(x.shape, (6, 8))


class TestInnModeGlow(_Timed):
    """Likelihood-based INN training with Glow on images - the
    train_flowmodels3.py case."""

    @classmethod
    def setUpClass(cls):
        cls.model = FlowModel(
            image_shape=(8, 8, 3),
            bijector="glow",
            glow_num_blocks=2,
            glow_steps_per_block=1,
            glow_num_hidden=8,
            y_dim=2,
            lambda_nll=192.0,
            lambda_y=10.0,
            lambda_z=0.0,
            lambda_x=0.0,
            lambda_pad=0.0,
            y_sigma=0.1,
        )
        cls.model.compile(optimizer="adam", jit_compile=False)
        cls.x = tf.random.uniform([2, 8, 8, 3])
        cls.y = tf.one_hot([0, 1], 2)
        _ = cls.model(cls.x)

    def test_train_step_runs(self):
        history = _one_train_step(self.model, self.x, self.y)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

    def test_forward_yz_shapes(self):
        y, z, pad = self.model.forward_yz(self.x.numpy())
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(z.shape, (2, 8 * 8 * 3 - 2))
        self.assertEqual(pad.shape, (2, 0))

    def test_sample_posterior_returns_images(self):
        imgs = self.model.sample_posterior([1.0, 0.0], 3, temperature=0.7)
        self.assertEqual(imgs.shape, (3, 8, 8, 3))


class TestLabeledGenerators(_Timed):

    def test_gmm_with_labels_and_padding(self):
        gen = get_data_generator(
            "gmm", batch_size=32, class_mode="categorical", pad_to=16
        )
        x, y = next(gen)
        self.assertEqual(x.shape, (32, 16))
        self.assertEqual(y.shape, (32, 4))
        np.testing.assert_allclose(y.sum(axis=1), 1.0)
        self.assertTrue(np.all(x[:, 2:] == 0))

    def test_gmm_unlabeled_unchanged(self):
        x = next(get_data_generator("gmm", batch_size=32))
        self.assertEqual(x.shape, (32, 2))

    def test_invkin_and_glacgrav(self):
        x, y = next(get_data_generator("invkin", batch_size=16, pad_to=4))
        self.assertEqual((x.shape, y.shape), ((16, 4), (16, 2)))
        x, y = next(get_data_generator("glacgrav", batch_size=16, pad_to=52))
        self.assertEqual((x.shape, y.shape), ((16, 52), (16, 12)))


if __name__ == "__main__":
    unittest.main()
