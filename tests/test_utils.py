"""
Tests of utils.py and non-S3 file_utils functionality, runnable on CPU
without GPU, MLflow, or image data.
"""

import time
import unittest
import numpy as np
from sklearn.decomposition import PCA

import file_utils
import utils


class TestDequantizeGenerator(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def _gen(self, arrays):
        for a in arrays:
            yield a

    def test_noise_is_added(self):
        base = np.zeros((4, 8, 8, 3), dtype=np.float32)
        result = next(file_utils.dequantize_generator(self._gen([base])))
        self.assertTrue(np.any(result != base))

    def test_noise_in_range(self):
        base = np.zeros((4, 8, 8, 3), dtype=np.float32)
        result = next(file_utils.dequantize_generator(self._gen([base])))
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result < 1.0 / 255))

    def test_dtype_preserved(self):
        base = np.zeros((4, 4, 4, 1), dtype=np.float32)
        result = next(file_utils.dequantize_generator(self._gen([base])))
        self.assertEqual(result.dtype, np.float32)

    def test_multiple_batches(self):
        arrays = [np.zeros((2, 4, 4, 3), dtype=np.float32) for _ in range(3)]
        results = list(file_utils.dequantize_generator(self._gen(arrays)))
        self.assertEqual(len(results), 3)


class TestInfiniteGenerator(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def _infinite_array_gen(self):
        while True:
            yield np.array([[1.0, 2.0]])

    def _infinite_tuple_gen(self):
        while True:
            yield (np.array([[1.0, 2.0]]), np.array([[0]]))

    def test_yields_multiple(self):
        gen = file_utils.infinite_generator(self._infinite_array_gen())
        results = [next(gen) for _ in range(5)]
        self.assertEqual(len(results), 5)

    def test_wraps_array_in_tuple(self):
        gen = file_utils.infinite_generator(self._infinite_array_gen())
        result = next(gen)
        self.assertIsInstance(result, tuple)

    def test_passes_through_existing_tuple(self):
        gen = file_utils.infinite_generator(self._infinite_tuple_gen())
        result = next(gen)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestGetDataGeneratorSynthetic(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_moons_batch_shape(self):
        gen = file_utils.get_data_generator("moons", batch_size=16)
        batch = next(gen)
        self.assertEqual(batch.shape, (16, 2))

    def test_gmm_batch_shape(self):
        gen = file_utils.get_data_generator("gmm", batch_size=32)
        batch = next(gen)
        self.assertEqual(batch.shape, (32, 2))

    def test_mvn_batch_shape(self):
        gen = file_utils.get_data_generator("mvn", batch_size=8)
        batch = next(gen)
        self.assertEqual(batch.shape, (8, 2))

    def test_moons_dtype(self):
        gen = file_utils.get_data_generator("moons", batch_size=4)
        batch = next(gen)
        self.assertEqual(batch.dtype, np.float32)

    def test_invalid_dataset_raises(self):
        with self.assertRaises(ValueError):
            file_utils.get_data_generator("nonexistent_dataset", batch_size=4)

    def test_cats_without_images_path_raises(self):
        with self.assertRaises(ValueError):
            gen = file_utils.get_data_generator("cats", batch_size=4)
            next(gen)


class TestInterpolateBetweenPoints(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_euclidean_output_shape(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = utils.interpolate_between_points(pts, N=5, path="euclidean")
        self.assertEqual(result.shape, (5, 2))

    def test_euclidean_endpoints(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = utils.interpolate_between_points(pts, N=4, path="euclidean")
        np.testing.assert_allclose(result[0], pts[0], atol=1e-5)
        np.testing.assert_allclose(result[-1], pts[1], atol=1e-5)

    def test_euclidean_midpoint(self):
        pts = np.array([[0.0, 0.0], [2.0, 2.0]])
        result = utils.interpolate_between_points(pts, N=3, path="euclidean")
        np.testing.assert_allclose(result[1], [1.0, 1.0], atol=1e-5)

    def test_slerp_output_shape(self):
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = utils.interpolate_between_points(pts, N=5, path="slerp")
        self.assertEqual(result.shape, (5, 3))

    def test_invalid_path_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            utils.interpolate_between_points(pts, N=3, path="invalid")


class TestGenerateMultivariateNormalSamples(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 10))
        self.pca = PCA(n_components=5)
        self.pca.fit(data)
        self.mean = np.zeros(10)
        self.cov = np.eye(5)

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_output_shape(self):
        samples = utils.generate_multivariate_normal_samples(
            self.mean, self.cov, self.pca, num_samples=7
        )
        self.assertEqual(samples.shape, (7, 10))

    def test_cov_scale_applied(self):
        samples_1 = utils.generate_multivariate_normal_samples(
            self.mean, self.cov, self.pca, num_samples=100, cov_scale=1.0
        )
        samples_small = utils.generate_multivariate_normal_samples(
            self.mean, self.cov, self.pca, num_samples=100, cov_scale=0.01
        )
        self.assertGreater(np.std(samples_1.numpy()), np.std(samples_small.numpy()))

    def test_none_pca_raises(self):
        with self.assertRaises(ValueError):
            utils.generate_multivariate_normal_samples(
                self.mean, self.cov, pca=None, num_samples=3
            )


if __name__ == "__main__":
    unittest.main()
