"""
Tests of flow_model functionality, runnable on CPU without GPU, MLflow, or image data.
Small synthetic tensors are used throughout to keep runtime short.
"""

import time
import unittest
import numpy as np
import tensorflow as tf

from flow_model import FlowModel, ShiftAndLogScaleDense


class TestShiftAndLogScaleDense(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_output_shape(self):
        layer = ShiftAndLogScaleDense(output_dim=4, hidden_layers=[8])
        out = layer(tf.random.normal([2, 4]))
        self.assertEqual(out.shape, (2, 8))  # 2 * output_dim

    def test_log_scale_clip_bounds(self):
        layer = ShiftAndLogScaleDense(output_dim=8, hidden_layers=[16], log_scale_clip=2.0)
        out = layer(tf.random.normal([10, 8]))
        _, log_scale = tf.split(out, 2, axis=-1)
        self.assertTrue(tf.reduce_all(log_scale >= -2.0).numpy())
        self.assertTrue(tf.reduce_all(log_scale <= 2.0).numpy())

    def test_no_clip_does_not_restrict(self):
        layer = ShiftAndLogScaleDense(output_dim=4, hidden_layers=[8], log_scale_clip=None)
        out = layer(tf.random.normal([5, 4]))
        self.assertEqual(out.shape, (5, 8))


class TestFlowModelRealNVP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_shape = (4, 4, 1)
        cls.flat_size = int(np.prod(cls.image_shape))
        cls.model = FlowModel(
            image_shape=cls.image_shape,
            bijector="realnvp-based",
            realnvp_flow_steps=2,
            realnvp_hidden_layers=[8],
            validate_args=False,
        )

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_call_output_shape(self):
        x = tf.random.normal([5, *self.image_shape])
        z = self.model.call(x)
        self.assertEqual(z.shape, (5, self.flat_size))

    def test_inverse_output_shape(self):
        z = tf.random.normal([5, self.flat_size])
        x = self.model.inverse(z)
        self.assertEqual(x.shape, (5, self.flat_size))

    def test_round_trip(self):
        x = tf.random.normal([3, *self.image_shape])
        z = self.model.call(x)
        x_hat = tf.reshape(self.model.inverse(z), x.shape)
        np.testing.assert_allclose(x.numpy(), x_hat.numpy(), atol=1e-4)

    def test_log_prob_finite(self):
        x = tf.reshape(tf.random.normal([4, *self.image_shape]), (-1, self.flat_size))
        log_prob = self.model.log_prob(x)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_prob)).numpy())

    def test_alternating_permutation_round_trip(self):
        model = FlowModel(
            image_shape=(4, 4, 1),
            bijector="realnvp-based",
            realnvp_flow_steps=4,
            realnvp_hidden_layers=[8],
            realnvp_permutation="alternating",
        )
        x = tf.random.normal([2, 4, 4, 1])
        z = model.call(x)
        x_hat = tf.reshape(model.inverse(z), x.shape)
        np.testing.assert_allclose(x.numpy(), x_hat.numpy(), atol=1e-4)

    def test_random_permutation_round_trip(self):
        model = FlowModel(
            image_shape=(4, 4, 1),
            bijector="realnvp-based",
            realnvp_flow_steps=4,
            realnvp_hidden_layers=[8],
            realnvp_permutation="random",
        )
        x = tf.random.normal([2, 4, 4, 1])
        z = model.call(x)
        x_hat = tf.reshape(model.inverse(z), x.shape)
        np.testing.assert_allclose(x.numpy(), x_hat.numpy(), atol=1e-4)


class TestFlowModelGlow(unittest.TestCase):
    # image_shape=(8,8,3): after squeeze, channels become 12, 24, ... all even — safe for Glow splits

    @classmethod
    def setUpClass(cls):
        cls.image_shape = (8, 8, 3)
        cls.flat_size = int(np.prod(cls.image_shape))
        cls.model = FlowModel(
            image_shape=cls.image_shape,
            bijector="glow",
            glow_num_blocks=2,
            glow_steps_per_block=1,
            glow_num_hidden=8,
            validate_args=False,
        )

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_call_output_shape(self):
        x = tf.random.normal([3, *self.image_shape])
        z = self.model.call(x)
        self.assertEqual(z.shape, (3, self.flat_size))

    def test_inverse_output_shape(self):
        z = tf.random.normal([3, self.flat_size])
        x = self.model.inverse(z)
        self.assertEqual(tuple(x.shape), (3, *self.image_shape))

    def test_round_trip(self):
        x = tf.random.normal([2, *self.image_shape])
        z = self.model.call(x)
        x_hat = self.model.inverse(z)
        np.testing.assert_allclose(x.numpy(), x_hat.numpy(), atol=1e-4)

    def test_log_prob_finite(self):
        x = tf.random.normal([2, *self.image_shape])
        log_prob = self.model.log_prob(x)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_prob)).numpy())


if __name__ == "__main__":
    unittest.main()
