import os
import random
import numpy
import multiprocessing.pool

import tensorflow as tf
import albumentations as A
import tensorflow.keras.backend as K

from datetime import datetime
from glob import glob
from dataloader.pillow_API import load_img
from dataloader.augmentations import which_custom_augmentations


def standardise(image):
    return (image - image.min()) / ((image.max() - image.min()) + K.epsilon())


def Compute_Mean_Std_Dataset(config, filenames):
    print(f"Computing mean and stddev over entire dataset ({len(filenames)} images)...")
    start_time = datetime.now()

    def get_mean_std(filepath):
        x = load_img(filepath, grayscale=False, target_size=(config["training.CroppedImageSize"][0],
                                                             config["training.CroppedImageSize"][1]), data_format=None)
        x = standardise(x)

        tmp_mean = numpy.mean(x)
        tmp_var = numpy.var(x)

        return tmp_mean, tmp_var, x.size

    def mean_std_unpack(args):
        _, filepath = args[0], args[1]
        return get_mean_std(filepath)

    pool = multiprocessing.pool.ThreadPool()
    samples = pool.map(mean_std_unpack, ((i, filepath) for i, filepath in enumerate(filenames)))

    mean = 0
    var = 0
    k = 0

    for batch_mean, batch_var, s in samples:
        old_mean = mean

        mean = (k * 1.0 / (k + s)) * mean + (s * 1.0 / (k + s)) * batch_mean
        var = (k * 1.0 / (k + s)) * var + \
              (s * 1.0 / (k + s)) * batch_var + \
              ((k * s) * 1.0 / ((k + s) * (k + s))) * ((old_mean - batch_mean) * (old_mean - batch_mean))

        k += s

    std = numpy.sqrt(var)
    print(f"\nTime Spent: {datetime.now() - start_time}")
    return mean, std


def Compute_Mean_Std_Dataset_RGB(config, filenames):
    start_time = datetime.now()

    def get_mean_std(filepath):
        x = load_img(filepath, grayscale=False, target_size=(config["training.CroppedImageSize"][0],
                                                             config["training.CroppedImageSize"][1]), data_format=None)
        x = standardise(x)

        if len(x.shape) == 3:
            tmp_sum = numpy.sum(x, axis=(0, 1), keepdims=False)
            tmp_sum_of_square = numpy.sum((x ** 2), axis=(0, 1), keepdims=False)

        if len(x.shape) == 4:
            tmp_sum = numpy.sum(x, axis=(0, 1, 2), keepdims=False)
            tmp_sum_of_square = numpy.sum((x ** 2), axis=(0, 1, 2), keepdims=False)

        return tmp_sum, tmp_sum_of_square, x.shape

    def mean_std_unpack(args):
        _, filepath = args[0], args[1]
        return get_mean_std(filepath)

    pool = multiprocessing.pool.ThreadPool()
    samples = pool.map(mean_std_unpack, ((i, filepath) for i, filepath in enumerate(filenames)))

    _sum_ = numpy.array([0.0, 0.0, 0.0])
    sum_of_square = numpy.array([0.0, 0.0, 0.0])
    for batch_sum, batch_sum_of_square, s in samples:
        _sum_ += batch_sum
        sum_of_square += batch_sum_of_square

    # pixel count
    count = config["training.NumExamples"] * s[0] * s[1]
    # Perform final calculations to obtain data-level mean and standard deviation
    mean = _sum_ / count
    variance = (sum_of_square / count) - (mean ** 2)
    std = numpy.sqrt(variance)
    print(f"\nTime Spent: {datetime.now() - start_time}")
    return mean, std


class TFDataLoader:
    def __init__(self, config, custom_augment=False, print_data_specifications=False):
        """
        - Using Command Line arguments
        """
        self.config = config
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.custom_augment = custom_augment
        self.print_data_specifications = print_data_specifications
        if self.custom_augment:
            self.transforms = A.Compose(which_custom_augmentations(config=self.config))

    def GetPNGImageFiles(self, num_examples_mode=False):
        folders = os.listdir(os.path.join(self.config["data.DataDir"]))
        filenames = glob(os.path.join(self.config["data.DataDir"], "*", "*.png"))
        random.shuffle(filenames)

        if self.print_data_specifications:
            print(f"Found {len(filenames)} images belonging to {len(folders)} classes.")
        if num_examples_mode:
            return len(filenames)
        else:
            return filenames

    def PILImagesRead(self, paths):
        images = load_img(paths, grayscale=False, target_size=(self.config["training.CroppedImageSize"][0],
                                                               self.config["training.CroppedImageSize"][1]),
                          data_format=None)
        return standardise(images)

    def ParseImages(self, paths):
        return tf.numpy_function(func=self.PILImagesRead, inp=[paths], Tout=tf.float32)

    def AugmentFunction(self, image):
        return image, self.transforms(**{"image": image})["image"], self.transforms(**{"image": image})["image"]

    def ProcessAugmentations(self, image):
        return tf.numpy_function(func=self.AugmentFunction, inp=[image], Tout=[tf.float32, tf.float32, tf.float32])

    def AugmentationReshape(self, images, augmented_images1, augmented_images2):
        augmented_images1.set_shape((self.config["training.CroppedImageSize"][0],
                                     self.config["training.CroppedImageSize"][1],
                                     self.config["training.ImageChannels"]))

        augmented_images2.set_shape((self.config["training.CroppedImageSize"][0],
                                     self.config["training.CroppedImageSize"][1],
                                     self.config["training.ImageChannels"]))

        return images, augmented_images1, augmented_images2

    def PerformanceConfiguration(self, ds):
        buffer_multiplier = 50 if self.config["training.CroppedImageSize"][0] <= 32 else 10
        ds = ds.shuffle(self.config["training.BatchSize"] * buffer_multiplier)
        ds = ds.batch(self.config["training.BatchSize"], drop_remainder=False)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def LoadDataset(self):
        filenames_ds = tf.data.Dataset.from_tensor_slices(self.GetPNGImageFiles(num_examples_mode=False))
        ds = filenames_ds.map(self.ParseImages, num_parallel_calls=self.AUTOTUNE)

        if self.custom_augment:
            ds = ds.map(self.ProcessAugmentations, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
            ds = ds.map(self.AugmentationReshape, num_parallel_calls=self.AUTOTUNE)

        ds = self.PerformanceConfiguration(ds)
        return ds

