"""
Tests that a single train step (forward + backward pass) runs without error
for each bijector type and problem shape. These exercise the gradient
computation path that forward-pass-only tests miss, including the
trainable_variables property, gradient clipping, and the optimizer update.
All tests run on CPU with tiny synthetic tensors; no GPU or image data needed.
"""

import time
import unittest
import numpy as np
import tensorflow as tf

from flow_model import FlowModel


def _one_train_step(model, dummy):
    # Use fit() on a single-batch dataset rather than train_on_batch():
    # Keras 3.x wraps data as (x, None) in train_on_batch, which breaks
    # our custom train_step signature. fit() with a dataset does not.
    ds = tf.data.Dataset.from_tensors(dummy)
    model.fit(ds, steps_per_epoch=1, epochs=1, verbose=0)


class TestTrainStepRealNVP2D(unittest.TestCase):
    """Train step for 2-D point data — matches the train_flowmodels1.py case."""

    @classmethod
    def setUpClass(cls):
        cls.image_shape = (2,)
        cls.model = FlowModel(
            image_shape=cls.image_shape,
            bijector="realnvp-based",
            realnvp_flow_steps=2,
            realnvp_hidden_layers=[8],
            validate_args=False,
        )
        cls.model.compile(optimizer="adam")
        cls.dummy = tf.random.normal([8, 2])
        # Forward pass initializes Keras layer variables (created lazily on
        # first call); without this, trainable_variables returns empty.
        _ = cls.model(cls.dummy)

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_trainable_variables_nonempty(self):
        self.assertGreater(len(self.model.trainable_variables), 0)

    def test_train_step_runs(self):
        _one_train_step(self.model, self.dummy)

    def test_log_prob_finite_after_train_step(self):
        _one_train_step(self.model, self.dummy)
        log_prob = self.model.log_prob(self.dummy)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_prob)).numpy())

    def test_train_step_with_grad_clipping(self):
        model = FlowModel(
            image_shape=(2,),
            bijector="realnvp-based",
            realnvp_flow_steps=2,
            realnvp_hidden_layers=[8],
            grad_norm_thresh=10.0,
            validate_args=False,
        )
        model.compile(optimizer="adam")
        _ = model(self.dummy)
        _one_train_step(model, self.dummy)


class TestTrainStepRealNVPImage(unittest.TestCase):
    """Train step for image-shaped input with the realnvp-based bijector."""

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
        cls.model.compile(optimizer="adam")
        cls.dummy = tf.random.normal([4, 4, 4, 1])
        _ = cls.model(cls.dummy)

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_trainable_variables_nonempty(self):
        self.assertGreater(len(self.model.trainable_variables), 0)

    def test_train_step_runs(self):
        _one_train_step(self.model, self.dummy)

    def test_log_prob_finite_after_train_step(self):
        _one_train_step(self.model, self.dummy)
        flat = tf.reshape(self.dummy, (-1, self.flat_size))
        log_prob = self.model.log_prob(flat)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_prob)).numpy())

    def test_train_step_with_grad_clipping(self):
        model = FlowModel(
            image_shape=(4, 4, 1),
            bijector="realnvp-based",
            realnvp_flow_steps=2,
            realnvp_hidden_layers=[8],
            grad_norm_thresh=10.0,
            validate_args=False,
        )
        model.compile(optimizer="adam")
        _ = model(self.dummy)
        _one_train_step(model, self.dummy)


class TestTrainStepGlow(unittest.TestCase):
    """Train step for the Glow bijector on a minimal spatial image tensor.

    image_shape=(8,8,3): after each squeeze, channel count is 12 then 48 —
    both divisible for Glow's split operations. jit_compile must be False
    (XLA incompatibility with Glow's 1x1 conv variables, same as production).
    """

    @classmethod
    def setUpClass(cls):
        cls.image_shape = (8, 8, 3)
        cls.model = FlowModel(
            image_shape=cls.image_shape,
            bijector="glow",
            glow_num_blocks=2,
            glow_steps_per_block=1,
            glow_num_hidden=8,
            validate_args=False,
        )
        cls.model.compile(optimizer="adam", jit_compile=False)
        cls.dummy = tf.random.normal([2, 8, 8, 3])
        _ = cls.model(cls.dummy)

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_trainable_variables_nonempty(self):
        self.assertGreater(len(self.model.trainable_variables), 0)

    def test_train_step_runs(self):
        _one_train_step(self.model, self.dummy)

    def test_log_prob_finite_after_train_step(self):
        _one_train_step(self.model, self.dummy)
        log_prob = self.model.log_prob(self.dummy)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_prob)).numpy())

    def test_train_step_with_grad_clipping(self):
        model = FlowModel(
            image_shape=(8, 8, 3),
            bijector="glow",
            glow_num_blocks=2,
            glow_steps_per_block=1,
            glow_num_hidden=8,
            grad_norm_thresh=10.0,
            validate_args=False,
        )
        model.compile(optimizer="adam", jit_compile=False)
        _ = model(self.dummy)
        _one_train_step(model, self.dummy)


if __name__ == "__main__":
    unittest.main()
