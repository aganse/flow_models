"""
Tests of gpu functionality.
"""

import time
import unittest

import tensorflow as tf


class TestGPU(unittest.TestCase):
    """
    Verify functionality of GPU
    """

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_gpu_matmul(self):
        # Ensure TensorFlow is using GPU
        tf.debugging.set_log_device_placement(True)

        # Define matrices to multiply
        size = 100
        matrix1 = tf.random.normal([size, size], dtype=tf.float32)
        matrix2 = tf.random.normal([size, size], dtype=tf.float32)

        # Repeat matrix multiplication 100 times
        for _ in range(100):
            result = tf.matmul(matrix1, matrix2)
            tf.reduce_sum(result).numpy()  # This forces the computation to happen


if __name__ == "__main__":
    unittest.main()
