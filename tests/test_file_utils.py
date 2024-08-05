"""
Tests of file_utils functionality.
"""

import os
import time
import unittest

import file_utils


class TestFileUtils(unittest.TestCase):
    """
    Verify functionality of classes/functions in file_utils.py
    """

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))

    def test_datagen_init(self):
        datagen = file_utils.S3ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            zoom_range=0.2,         # currently still debugging for zoom_range=tuple like (0.8, 1.2)
            shear_range=0.0,        # currently still debugging for shear_range!=0.0
            rotation_range=30,
            width_shift_range=0.0,  # currently still debugging for width_shift_range!=0.0
            height_shift_range=0.0  # currently still debugging for height_shift_range!=0.0
        )
        self.assertEqual(type(datagen), file_utils.S3ImageDataGenerator)

    def test_datagen_flow(self):
        datagen = file_utils.S3ImageDataGenerator()
        train_generator = datagen.flow_from_directory(
            os.environ["FLOW_MODELS_S3URI"],
            target_size=(128, 128),
            batch_size=32,
            class_mode=None,
            shuffle=True,
        )
        self.assertEqual(next(train_generator).shape, (32, 128, 128, 3))


if __name__ == "__main__":
    unittest.main()
