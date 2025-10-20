import tensorflow as tf
import cv2
import numpy as np
import tensorflow.keras.backend as K


def standardise_sample(sample):
    # return (sample - tf.reduce_min(sample)) / ((tf.reduce_max(sample) - tf.reduce_min(sample)) + K.epsilon())
    return (sample - sample.min()) / ((sample.max() - sample.min()) + K.epsilon())


def normalise_sample(sample, mean, stddev):
    sample -= mean
    sample /= stddev

    return sample


def normalize_sample_albumentation(sample, mean, stddev, max_pixel_value=1.0):
    def normalize_cv2(img, mean, denominator):
        if mean.shape and len(mean) != 4 and mean.shape != img.shape:
            mean = np.array(mean.tolist() + [0] * (4 - len(mean)), dtype=np.float64)
        if not denominator.shape:
            denominator = np.array([denominator.tolist()] * 4, dtype=np.float64)
        elif len(denominator) != 4 and denominator.shape != img.shape:
            denominator = np.array(denominator.tolist() + [1] * (4 - len(denominator)), dtype=np.float64)

        img = np.ascontiguousarray(img.astype("float32"))
        cv2.subtract(img, mean.astype(np.float64), img)
        cv2.multiply(img, denominator.astype(np.float64), img)
        return img

    def normalize_numpy(img, mean, denominator):
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    def normalize(img, mean, std, max_pixel_value=1.0):
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value

        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        if img.ndim == 3 and img.shape[-1] == 3:
            return normalize_cv2(img, mean, denominator)

        return normalize_numpy(img, mean, denominator)

    sample = normalize(sample, mean, stddev, max_pixel_value)
    return sample
