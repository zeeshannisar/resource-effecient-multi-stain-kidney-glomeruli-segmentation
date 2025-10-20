from __future__ import absolute_import
from __future__ import print_function

from augmentation import smote, elastic_transform, affine_transform, stain_transform, blur_transform, noise_transform, enhance_transform, channel_transform, stain_transfer_transform

import numpy
import re
import h5py
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial
import random
import tensorflow as tf

### Zeeshan Change
from datetime import datetime
####

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from utils import data_utils, image_utils

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = numpy.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-numpy.min(x), 0)
        x_max = numpy.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = numpy.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def crop_image(img, target_size, data_format):
    if data_format == 'channels_first':
        img_size = img.shape[1:]
    elif data_format:
        img_size = img.shape[:-1]

    if img_size[0] < target_size[0] or img_size[1] < target_size[1]:
        raise ValueError('Invalid cropped image size (%s). Image is %d x %d and target size is %d x %d.' % (
        path, img_size[0], img_size[1], target_size[0], target_size[1]))

    if (img_size[0] - target_size[0]) % 2 != 0:
        raise ValueError(
            'Invalid cropped image size. There should be an even difference between the image and target heights')

    if (img_size[1] - target_size[1]) % 2 != 0:
        raise ValueError(
            'Invalid cropped image size. There should be an even difference between the image and target widths')

    if img_size != target_size:
        diffs = numpy.subtract(img_size, target_size)
        diffs //= 2

        img = img[diffs[0]:img_size[0] - diffs[0], diffs[1]:img_size[1] - diffs[1]]

    return img


def load_img(path, grayscale=False, target_size=None, data_format=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    img = pil_image.open(path)

    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')

    img = img_to_array(img, data_format=data_format)

    if target_size is not None:
        img = crop_image(img, target_size)

    return img


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        categoricaltarget: whether the target should be converted to one-hot encoding
    """

    def __init__(self,
                 nb_classes=2,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 samplewise_normalise=False,
                 standardise_sample=False,
                 methods=[],
                 augmentationparameters=None,
                 fill_mode='reflect',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 categoricaltarget=True,
                 validation_split=None):

        if data_format is None:
            data_format = K.image_data_format()

        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.samplewise_normalise = samplewise_normalise
        self.standardise_sample = standardise_sample
        self.methods = methods
        self.augmentationparameters = augmentationparameters
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.nb_classes = nb_classes
        self.categoricaltarget = categoricaltarget

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format should be "channels_last" (channel after row and '
                             'column) or "channels_first" (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        if validation_split and not 0 < validation_split < 1:
            raise ValueError('`validation_split` must be strictly between 0 and 1. '
                             ' Received arg: ', validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.class_weights = None
        self.principal_components = None
        self.dataset_mean = None
        self.dataset_std = None
        self.number_of_validation_samples = None

    def flow(self, x, y=None, staincodes=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', stainsubdir=None):
        if 'stain' in self.methods and not staincodes:
            raise ValueError('A list of stain codes for each image must be given when using stain augmentation')

        if 'stain_transfer' in self.methods:
            if stainsubdir:
                self.stainsubdir = stainsubdir
            else:
                raise ValueError('A stain subdirectory must be given when using stain augmentation')

        return NumpyArrayIterator(
            x, y, self,
            staincodes=staincodes,
            nb_classes=self.nb_classes,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            categoricaltarget=self.categoricaltarget)

    def flow_from_directory(self, directory,
                            filepath=None,
                            img_target_size=None,
                            gt_target_size=None,
                            color_mode='rgb',
                            classes=None,
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            dataset_mean=None,
                            dataset_std=None,
                            augmentationclassblock={},
                            pretrained_models=None):
        """Takes the path to a directory, and generates batches of augmented/normalized data.
        # Arguments
                directory: path to the target directory.
                 It should contain one subdirectory per class.
                 Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator.
                See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d) for more details.
                img_target_size: tuple of integers `(height, width)`.
                 The dimensions to which all images found will be cropped.
                gt_target_size: tuple of integers `(height, width)`.
                 The dimensions to which all GTs found will be cropped.
                color_mode: one of "greyscale", "rbg". Default: "rgb".
                 Whether the images will be converted to have 1 or 3 color channels.
                classes: optional list of class subdirectories (e.g. `['dogs', 'cats']`).
                 Default: None. If not provided, the list of classes will
                 be automatically inferred from the subdirectory names/structure under `directory`,
                 where each subdirectory will be treated as a different class
                 (and the order of the classes, which will map to the label indices, will be alphanumeric).
                 The dictionary containing the mapping from class names to class
                 indices can be obtained via the attribute `class_indices`.
                class_mode: one of "categorical", "binary", "sparse", "input" or None.
                 Default: "categorical". Determines the type of label arrays that are
                 returned: "categorical" will be 2D one-hot encoded labels, "binary" will be 1D binary labels,
                 "sparse" will be 1D integer labels, "input" will be images identical to input images (mainly used to work with autoencoders).
                 If None, no labels are returned (the generator will only yield batches of image data, which is useful to use
                 `model.predict_generator()`, `model.evaluate_generator()`, etc.).
                  Please note that in case of class_mode None,
                   the data still needs to reside in a subdirectory of `directory` for it to work correctly.
                batch_size: size of the batches of data (default: 32).
                shuffle: whether to shuffle the data (default: True)
                seed: optional random seed for shuffling and transformations.
                save_to_dir: None or str (default: None). This allows you to optionally specify a directory to which to save
                 the augmented pictures being generated (useful for visualizing what you are doing).
                save_prefix: str. Prefix to use for filenames of saved pictures (only relevant if `save_to_dir` is set).
                save_format: one of "png", "jpeg" (only relevant if `save_to_dir` is set). Default: "png".
                follow_links: whether to follow symlinks inside class subdirectories (default: False).
        # Returns
            A DirectoryIterator yielding tuples of `(x, y)` where `x` is a numpy array of image data and
             `y` is a numpy array of corresponding labels.
        """
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        if 'stain' in self.methods and not filepath:
            raise ValueError('A filepath object must be given when using stain augmentation')

        if 'stain_transfer' in self.methods:
            if directory.endswith(os.sep):
                directory = directory[:-1]

            if directory.endswith('validation'):
                self.stainsubdir = 'validation'
            elif directory.endswith('train'):
                self.stainsubdir = 'train'
            else:
                raise ValueError('Stain transfer target subdir not recognised')

        return DirectoryIterator(directory, self,
                                 filepath=filepath,
                                 img_target_size=img_target_size,
                                 gt_target_size=gt_target_size,
                                 color_mode=color_mode,
                                 classes=classes,
                                 data_format=self.data_format,
                                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                                 save_to_dir=save_to_dir,
                                 save_prefix=save_prefix,
                                 save_format=save_format,
                                 follow_links=follow_links,
                                 subset=subset,
                                 categoricaltarget=self.categoricaltarget,
                                 augmentationclassblock=augmentationclassblock,
                                 pretrained_models=pretrained_models)

    def fit_and_flow_from_directory(self, directory,
                                    filepath=None,
                                    img_target_size=None,
                                    gt_target_size=None,
                                    color_mode='rgb',
                                    classes=None,
                                    batch_size=32, shuffle=True, seed=None,
                                    save_to_dir=None,
                                    save_prefix='',
                                    save_format='png',
                                    follow_links=False,
                                    subset=None,
                                    augmentationclassblock={},
                                    pretrained_models=None):
        """Takes the path to a directory, and generates batches of augmented/normalized data.
        # Arguments
                directory: path to the target directory.
                 It should contain one subdirectory per class.
                 Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator.
                See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d) for more details.
                img_target_size: tuple of integers `(height, width)`.
                 The dimensions to which all images found will be cropped.
                gt_target_size: tuple of integers `(height, width)`.
                 The dimensions to which all GTs found will be cropped.
                color_mode: one of "greyscale", "rbg". Default: "rgb".
                 Whether the images will be converted to have 1 or 3 color channels.
                classes: optional list of class subdirectories (e.g. `['dogs', 'cats']`).
                 Default: None. If not provided, the list of classes will
                 be automatically inferred from the subdirectory names/structure under `directory`,
                 where each subdirectory will be treated as a different class
                 (and the order of the classes, which will map to the label indices, will be alphanumeric).
                 The dictionary containing the mapping from class names to class
                 indices can be obtained via the attribute `class_indices`.
                class_mode: one of "categorical", "binary", "sparse", "input" or None.
                 Default: "categorical". Determines the type of label arrays that are
                 returned: "categorical" will be 2D one-hot encoded labels, "binary" will be 1D binary labels,
                 "sparse" will be 1D integer labels, "input" will be images identical to input images (mainly used to work with autoencoders).
                 If None, no labels are returned (the generator will only yield batches of image data, which is useful to use
                 `model.predict_generator()`, `model.evaluate_generator()`, etc.).
                  Please note that in case of class_mode None,
                   the data still needs to reside in a subdirectory of `directory` for it to work correctly.
                batch_size: size of the batches of data (default: 32).
                shuffle: whether to shuffle the data (default: True)
                seed: optional random seed for shuffling and transformations.
                save_to_dir: None or str (default: None). This allows you to optionally specify a directory to which to save
                 the augmented pictures being generated (useful for visualizing what you are doing).
                save_prefix: str. Prefix to use for filenames of saved pictures (only relevant if `save_to_dir` is set).
                save_format: one of "png", "jpeg" (only relevant if `save_to_dir` is set). Default: "png".
                follow_links: whether to follow symlinks inside class subdirectories (default: False).
        # Returns
            A DirectoryIterator yielding tuples of `(x, y)` where `x` is a numpy array of image data and
             `y` is a numpy array of corresponding labels.
        """

        if 'stain' in self.methods and not filepath:
            raise ValueError('A filepath object must be given when using stain augmentation')

        if 'stain_transfer' in self.methods:
            if directory.endswith(os.sep):
                directory = directory[:-1]

            if directory.endswith('validation'):
                self.stainsubdir = 'validation'
            elif directory.endswith('train'):
                self.stainsubdir = 'train'
            else:
                raise ValueError('Stain transfer target subdir not recognised')

        if 'stain' in self.methods and 'stain_transfer' in self.methods:
            warnings.warn('Using ''stain'' augmentation with the stain_transfer training strategy is not currently '
                          'advised as the resulting normalisation statistics may not match realistic values')

        iterator = DirectoryIterator(
            directory, self,
            filepath=filepath,
            img_target_size=img_target_size,
            gt_target_size=gt_target_size,
            color_mode=color_mode,
            classes=classes,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            categoricaltarget=self.categoricaltarget,
            augmentationclassblock=augmentationclassblock,
            pretrained_models=pretrained_models)

        start = datetime.now()
        print('Calculating dataset-statistics (mean/std) over whole data...')
        self.dataset_mean, self.dataset_std, self.class_weights = iterator.get_fit_stats(self.standardise_sample)
        print('\nTime spent: {}'.format(datetime.now() - start))


        return iterator

    def get_fit_stats(self):
        if self.dataset_mean is not None and self.dataset_std is not None:
            return self.dataset_mean, self.dataset_std
        else:
            warnings.warn('This ImageDataGenerator hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)` or `fit_and_flow_from_directory(...)`.')

    def get_weights(self):

        if self.class_weights is not None:
            return self.class_weights
        else:
            warnings.warn('This ImageDataGenerator hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)` or `fit_and_flow_from_directory(...)`.')

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= numpy.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (numpy.std(x, axis=img_channel_axis, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = numpy.reshape(x, (x.size))
                whitex = numpy.dot(flatx, self.principal_components)
                x = numpy.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        if self.standardise_sample:
            x = data_utils.standardise_sample(x)

        if self.samplewise_normalise:
            if self.dataset_mean is not None and self.dataset_std is not None:
                x = data_utils.normalise_sample(x, self.dataset_mean, self.dataset_std)
            else:
                warnings.warn('This ImageDataGenerator specifies `samplewise_normalise`, but it hasn\'t'
                              'been fit on any training data. Fit it first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, y, j=None, staincode=None, block=[]):
        """Randomly augment a single image tensor + image mask.

        # Arguments
            x: 3D tensor, single image.
            y: 3D tensor, image mask.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1

        ##
        # Pick random transform from methods
        ##

        # if len(self.methods) > 0:
        #     methodindex = random.randint(0, len(self.methods) - 1)
        #
        #     if self.methods[methodindex] == 'affine':
        #
        #         return affine.transform(x[j].astype(K.floatx()), y[j].astype(K.floatx()),
        #                                 rotation_range=self.augmentationparameters['affine_rotation_range'],
        #                                 width_shift_range=self.augmentationparameters['affine_width_shift_range'],
        #                                 height_shift_range=self.augmentationparameters['affine_height_shift_range'],
        #                                 rescale=self.augmentationparameters['affine_rescale'],
        #                                 zoom_range=self.augmentationparameters['affine_zoom_range'],
        #                                 horizontal_flip=self.augmentationparameters['affine_horizontal_flip'],
        #                                 vertical_flip=self.augmentationparameters['affine_vertical_flip'],
        #                                 fill_mode=self.fill_mode,
        #                                 cval=self.cval)
        #
        #     if self.methods[methodindex] == 'smote':
        #
        #         return smote.generate_sample(x.astype(K.floatx()), y.astype(K.floatx()), j,
        #                               self.augmentationparameters['smotenneighbours'])
        #
        #     if self.methods[methodindex] == 'elastic':
        #
        #         return elastic_transform.transform(x[j].astype(K.floatx()), y[j].astype(K.floatx()),
        #                                            self.augmentationparameters['elastic_sigma'],
        #                                            self.augmentationparameters['elastic_alpha'])
        # else:
        #     return x[j], y[j]

        ##
        # Apply all transforms
        ##

        if j:
            x = x[j].astype(K.floatx())
            y = y[j].astype(K.floatx())
            if 'stain' in self.methods:
                staincode = staincode[j]
        else:
            x = x.astype(K.floatx())
            y = y.astype(K.floatx())

        ord = random.sample(range(len(self.methods)), len(self.methods))

        for i in ord:

            if 'elastic' not in block and self.methods[i] == 'elastic' and random.random() < 0.5:
                x, y = elastic_transform.transform(x, y,
                                                   sigma=self.augmentationparameters['elastic_sigma'],
                                                   alpha=self.augmentationparameters['elastic_alpha'])

            if 'affine' not in block and self.methods[i] == 'affine' and random.random() < 0.5:
                x, y = affine_transform.transform(x, y,
                                                  rotation_range=self.augmentationparameters['affine_rotation_range'],
                                                  width_shift_range=self.augmentationparameters['affine_width_shift_range'],
                                                  height_shift_range=self.augmentationparameters['affine_height_shift_range'],
                                                  rescale=self.augmentationparameters['affine_rescale'],
                                                  zoom_range=self.augmentationparameters['affine_zoom_range'],
                                                  horizontal_flip=self.augmentationparameters['affine_horizontal_flip'],
                                                  vertical_flip=self.augmentationparameters['affine_vertical_flip'],
                                                  fill_mode=self.fill_mode,
                                                  cval=self.cval)

            if 'stain' not in block and self.methods[i] == 'stain' and 'stain_transfer' not in self.methods and random.random() < 0.5:
                x, y = stain_transform.transform(x, y, staincode=staincode,
                                                 alpha_range=self.augmentationparameters['stain_alpha_range'],
                                                 beta_range=self.augmentationparameters['stain_beta_range'])

            if 'stain_transfer' not in block and self.methods[i] == 'stain_transfer':
                if not staincode in self.augmentationparameters['colour_transfer_targets'] or \
                        (staincode in self.augmentationparameters['colour_transfer_targets'] and
                         random.random() > 1 / (len(self.augmentationparameters['colour_transfer_targets']) + 1)):
                    alpha_range = None
                    if 'stain' in self.methods:
                        alpha_range = self.augmentationparameters['stain_alpha_range']
                    beta_range = None
                    if 'stain' in self.methods:
                        beta_range = self.augmentationparameters['stain_beta_range']
                    x, y = stain_transfer_transform.transform(x, y,
                                                              self.augmentationparameters['colour_transfer_staindatadir'],
                                                              self.stainsubdir,
                                                              self.augmentationparameters['colour_transfer_targets'],
                                                              'stain' in self.methods,
                                                              alpha_range=alpha_range,
                                                              beta_range=beta_range)

            if 'blur' not in block and self.methods[i] == 'blur' and random.random() < 0.5:
                x, y = blur_transform.transform(x, y,
                                                sigma_range=self.augmentationparameters['blur_sigma_range'])

            if 'noise' not in block and self.methods[i] == 'noise' and random.random() < 0.5:
                x, y = noise_transform.transform(x, y,
                                                 sigma_range=self.augmentationparameters['noise_sigma_range'])

            if 'enhance' not in block and self.methods[i] == 'enhance' and random.random() < 0.5:
                x, y = enhance_transform.transform(x, y,
                                                   bright_factor_range=self.augmentationparameters['bright_factor_range'],
                                                   contrast_factor_range=self.augmentationparameters['contrast_factor_range'],
                                                   colour_factor_range=self.augmentationparameters['colour_factor_range'])

            if 'channel' not in block and self.methods[i] == 'channel' and random.random() < 0.5:
                x, y = channel_transform.transform(x, y)

        if len(x.shape) == 2:
            x = x[..., None]

        if len(y.shape) == 2:
            y = y[..., None]

        return x, y

    def fit(self, x, y,
            augment=False,
            rounds=1,
            seed=None,
            staincodes=None,
            stainsubdir=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of greyscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            y: Numpy array, the ground truth of the data. All but the
                last axis should be the same as x.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = numpy.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if 'stain' in self.methods and 'stain_transfer' in self.methods:
            warnings.warn('Using ''stain'' augmentation with the stain_transfer training strategy is not currently '
                          'advised as the resulting normalisation statistics may not match realistic values')

        if 'stain' in self.methods and not staincodes:
            raise ValueError('A list of stain codes for each image must be given when using stain augmentation')

        if 'stain_transfer' in self.methods:
            if stainsubdir:
                self.stainsubdir = stainsubdir
            else:
                raise ValueError('A stain subdirectory must be given when using stain augmentation')

        if seed is not None:
            numpy.random.seed(seed)

        classes, counts = numpy.unique(y.astype(numpy.uint), return_counts=True)
        class_weights = numpy.zeros(self.nb_classes, dtype=numpy.uint)
        for cl, c in zip(classes, counts):
            class_weights[cl] += c
        #class_weights = (numpy.ones(self.nb_classes) * numpy.max(class_weights)) / class_weights
        #self.class_weights = class_weights / numpy.sum(class_weights)
        self.class_weights = 1 - (class_weights / class_weights.sum())

        x = numpy.copy(x)
        if augment:
            ax = numpy.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    if staincodes:
                        ax[i + r * x.shape[0]], _ = self.random_transform(x[i], x[i], staincode=staincodes[i])
                    else:
                        ax[i + r * x.shape[0]], _ = self.random_transform(x[i], x[i])

            x = ax

        if self.standardise_sample:
            for idx, sample in enumerate(x):
                x[idx, ] = data_utils.standardise_sample(sample)

        if self.samplewise_normalise:
            self.dataset_mean = numpy.mean(x)
            self.dataset_std = numpy.std(x)

        if self.featurewise_center:
            self.mean = numpy.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = numpy.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = numpy.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = numpy.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = numpy.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = numpy.dot(numpy.dot(u, numpy.diag(1. / numpy.sqrt(s + 10e-7))), u.T)


class Iterator(Sequence):
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed, nb_classes=2):
        self.n = n
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = numpy.arange(self.n)
        if self.shuffle:
            self.index_array = numpy.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx, length=len(self)))
        if self.seed is not None:
            numpy.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        # Zeeshan made changes here
        iterations_per_epoch = (self.n + self.batch_size - 1) // self.batch_size  # round up
        return iterations_per_epoch
        #
    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                numpy.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator, staincodes=None, nb_classes=2,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 categoricaltarget=True):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (numpy.asarray(x).shape, numpy.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = numpy.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn('NumpyArrayIterator is set to use the '
                             'data format convention "' + data_format + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = numpy.asarray(y, dtype=K.floatx())
        else:
            self.y = None
        self.staincodes = staincodes
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.categoricaltarget = categoricaltarget

        if save_to_dir and not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)

        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed, nb_classes=nb_classes)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x_shape = (len(index_array),) + self.x.shape[1:]
        batch_x = numpy.zeros(batch_x_shape, dtype=K.floatx())

        if self.categoricaltarget:
            batch_y = numpy.zeros(batch_x_shape[:3] + (self.nb_classes,), dtype=K.floatx())
        else:
            batch_y = numpy.zeros(batch_x_shape[:3] + (1,), dtype=K.floatx())

        for i, j in enumerate(index_array):
            x, y = self.image_data_generator.random_transform(self.x, self.y, j=j, staincode=self.staincodes)
            x = self.image_data_generator.standardize(x)

            if self.categoricaltarget:
                y = to_categorical(y, num_classes=self.nb_classes)

            batch_x[i] = x.astype(K.floatx())
            batch_y[i] = y.astype(K.floatx())

        if self.save_to_dir:
            for i in range(len(index_array)):
                imgx = array_to_img(batch_x[i], self.data_format, scale=True)
                if self.categoricaltarget:
                    tmpy = numpy.expand_dims(numpy.argmax(batch_y[i], axis=-1), axis=len(y.shape))
                else:
                    tmpy = batch_y[i]
                imgy = array_to_img(tmpy*(255/(self.nb_classes-1)), self.data_format, scale=False)
                fname = '{prefix}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                          hash=numpy.random.randint(1e4),
                                                          format=self.save_format)
                imgx.save(os.path.join(self.save_to_dir, 'x_' + fname))
                imgy.save(os.path.join(self.save_to_dir, 'y_' + fname))

        if self.y is None:
            return batch_x

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Count files with extension in `white_list_formats` contained in directory.
    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        follow_links: boolean.
    # Yields
        tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname


def _count_valid_files_in_directory(directory, white_list_formats, split, follow_links):
    """Count files with extension in `white_list_formats` contained in directory.
    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        follow_links: boolean.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    num_files = len(list(_iter_valid_files(directory, white_list_formats, follow_links)))
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start


def _list_valid_filenames_in_directory(directory, white_list_formats, split, class_indices, follow_links):
    """List paths of files in `subdir` with extensions in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean.
    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(_iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(_iter_valid_files(directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(directory, white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames


class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        img_target_size: tuple of integers, dimensions to crop input images to.
        gt_target_size: tuple of integers, dimensions to crop GTs images to.
        color_mode: One of `"rgb"`, `"greyscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator, filepath=None,
                 img_target_size=None, gt_target_size=None, color_mode='rgb',
                 classes=None,
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 categoricaltarget=True,
                 augmentationclassblock={},
                 pretrained_models=None):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.filepath = filepath
        if img_target_size:
            self.img_target_size = tuple(img_target_size)
        if gt_target_size:
            self.gt_target_size = tuple(gt_target_size)
        if color_mode not in {'rgb', 'greyscale', 'haemotoxylin'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "greyscale", or "haemotoxylin".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.img_image_shape = self.img_target_size + (3,)
            else:
                self.img_image_shape = (3,) + self.img_target_size
        else:
            if self.data_format == 'channels_last':
                self.img_image_shape = self.img_target_size + (1,)
            else:
                self.img_image_shape = (1,) + self.img_target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.augmentationclassblock = augmentationclassblock
        self.pretrained_models = pretrained_models

        if save_to_dir and not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(os.path.join(directory, 'images'))):
                if os.path.isdir(os.path.join(directory, 'images', subdir)):
                    classes.append(subdir)
        nb_classes = len(classes)

        self.categoricaltarget = categoricaltarget
        if self.categoricaltarget:
            if self.data_format == 'channels_last':
                self.gt_image_shape = self.gt_target_size + (nb_classes,)
            else:
                self.gt_image_shape = (nb_classes,) + self.gt_target_size
        else:
            if self.data_format == 'channels_last':
                self.gt_image_shape = self.gt_target_size + (1,)
            else:
                self.gt_image_shape = (1,) + self.gt_target_size

        self.classnames = classes

        self.class_indices = dict(zip(self.classnames, range(nb_classes)))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   split=split,
                                   follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, 'images', subdir)
                                     for subdir in self.classnames)))

        ### Zeeshan made some changes Here
        ### Start
        check_mode = directory.rsplit('/', 1)[1]
        if check_mode == 'train':
            final_mode = 'Training'
        elif check_mode == 'validation':
            final_mode = 'Validation'
        ### End

        print('\nFound %d %s images belonging to %d classes.' % (self.samples, final_mode, nb_classes))

        # second, build an index of the images in the different class subfolders
        results = []
        self.filenames = []
        self.class_weights = None
        self.classes = numpy.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, 'images', subdir) for subdir in self.classnames):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats, split,
                                             self.class_indices, follow_links)))

        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)

        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed, nb_classes=nb_classes)

    def get_fit_stats_old(self, standardise_samples):
        return 0.881942980766296, 0.10044879824069028, numpy.zeros(self.nb_classes, dtype=numpy.uint)

    def get_fit_stats(self, standardise_samples):

        def get_mean_std(filename, patchclass):
            x = load_img(os.path.join(self.directory, 'images', filename), data_format=self.data_format)

            y = load_img(os.path.join(self.directory, 'gts', filename),
                         grayscale=True, data_format=self.data_format).astype(numpy.uint)

            blockedaugmentations = []
            if patchclass in self.augmentationclassblock:
                blockedaugmentations = self.augmentationclassblock[patchclass]

            if self.filepath:
                staincode = self.filepath.get_stain(filename)
                x, y = self.image_data_generator.random_transform(x, y, staincode=staincode, block=blockedaugmentations)
            else:
                x, y = self.image_data_generator.random_transform(x, y, block=blockedaugmentations)

            x = crop_image(x, self.img_target_size, data_format=self.data_format)
            y = crop_image(y, self.gt_target_size, data_format=self.data_format)

            x = image_utils.image_colour_convert(x, self.color_mode)

            if standardise_samples:
                x = data_utils.standardise_sample(x)

            class_weights = numpy.zeros(self.nb_classes, dtype=numpy.uint)
            classes, counts = numpy.unique(y.astype(int), return_counts=True)
            class_weights[classes] += counts.astype(numpy.uint)

            batch_mean = numpy.mean(x)
            batch_var = numpy.var(x)



            return batch_mean, batch_var, x.size, class_weights

        def mean_std_unpack(args):
            return get_mean_std(args[0], args[1])

        pool = multiprocessing.pool.ThreadPool()
        samples = pool.map(mean_std_unpack,
                           ((file, self.classnames[self.classes[i]]) for i, file in enumerate(self.filenames)))

        mean = 0
        var = 0
        k = 0
        class_weights = numpy.zeros(self.nb_classes, dtype=numpy.uint)
        for batch_mean, batch_var, s, sample_class_weights in samples:
            old_mean = mean

            mean = (k * 1.0 / (k + s)) * mean + (s * 1.0 / (k + s)) * batch_mean
            var = (k * 1.0 / (k + s)) * var + \
                  (s * 1.0 / (k + s)) * batch_var + \
                  ((k * s) * 1.0 / ((k + s) * (k + s))) * ((old_mean - batch_mean) * (old_mean - batch_mean))

            k += s

            class_weights += sample_class_weights

        std = numpy.sqrt(var)

        #class_weights = (numpy.ones(self.nb_classes) * numpy.max(class_weights)) / class_weights
        #class_weights /= numpy.sum(class_weights)
        class_weights = 1 - (class_weights / class_weights.sum())

        return mean, std, class_weights

    def get_fit_stats_RGB(self, standardise_samples):
        def get_mean_std(filename, patchclass):
            x = load_img(os.path.join(self.directory, 'images', filename), data_format=self.data_format)

            y = load_img(os.path.join(self.directory, 'gts', filename),
                         grayscale=True, data_format=self.data_format).astype(numpy.uint)

            blockedaugmentations = []
            if patchclass in self.augmentationclassblock:
                blockedaugmentations = self.augmentationclassblock[patchclass]

            if self.filepath:
                staincode = self.filepath.get_stain(filename)
                x, y = self.image_data_generator.random_transform(x, y, staincode=staincode, block=blockedaugmentations)
            else:
                x, y = self.image_data_generator.random_transform(x, y, block=blockedaugmentations)

            x = crop_image(x, self.img_target_size, data_format=self.data_format)
            y = crop_image(y, self.gt_target_size, data_format=self.data_format)

            x = image_utils.image_colour_convert(x, self.color_mode)

            if standardise_samples:
                x = data_utils.standardise_sample(x)

            class_weights = numpy.zeros(self.nb_classes, dtype=numpy.uint)
            classes, counts = numpy.unique(y.astype(int), return_counts=True)
            class_weights[classes] += counts.astype(numpy.uint)

            if len(x.shape) == 3:
                batch_sum = numpy.sum(x, axis=(0, 1), keepdims=False)
                batch_sum_of_square = numpy.sum((x ** 2), axis=(0, 1), keepdims=False)

            if len(x.shape) == 4:
                batch_sum = numpy.sum(x, axis=(0, 1, 2), keepdims=False)
                batch_sum_of_square = numpy.sum((x ** 2), axis=(0, 1, 2), keepdims=False)

            return batch_sum, batch_sum_of_square, x.shape, class_weights

        def mean_std_unpack(args):
            return get_mean_std(args[0], args[1])

        pool = multiprocessing.pool.ThreadPool()
        samples = pool.map(mean_std_unpack,
                           ((file, self.classnames[self.classes[i]]) for i, file in enumerate(self.filenames)))

        _sum_ = numpy.array([0.0, 0.0, 0.0])
        sum_of_square = numpy.array([0.0, 0.0, 0.0])
        num_examples = 0
        class_weights = numpy.zeros(self.nb_classes, dtype=numpy.uint)
        for batch_sum, batch_sum_of_square, s, sample_class_weights in samples:
            num_examples += 1
            _sum_ += batch_sum
            sum_of_square += batch_sum_of_square
            class_weights += sample_class_weights

        # pixel count
        count = num_examples * s[0] * s[1]
        # Perform final calculations to obtain data-level mean and standard deviation
        mean = _sum_ / count
        variance = (sum_of_square / count) - (mean ** 2)
        std = numpy.sqrt(variance)
        class_weights = 1 - (class_weights / class_weights.sum())

        return mean.tolist(), std.tolist(), class_weights

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x_shape = (len(index_array),) + self.img_image_shape
        batch_x = numpy.zeros(batch_x_shape, dtype=K.floatx())

        batch_y_shape = (len(index_array),) + self.gt_image_shape
        batch_y = numpy.zeros(batch_y_shape, dtype=K.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            patchclass = self.classnames[self.classes[j]]
            x = load_img(os.path.join(self.directory, 'images', fname), grayscale=False, data_format=self.data_format)
            y = load_img(os.path.join(self.directory, 'gts', fname), grayscale=True, data_format=self.data_format)

            blockedaugmentations = []

            if patchclass in self.augmentationclassblock:
                blockedaugmentations = self.augmentationclassblock[patchclass]

            if self.filepath:
                staincode = self.filepath.get_stain(fname)
                x, y = self.image_data_generator.random_transform(x, y, staincode=staincode, block=blockedaugmentations)
            else:
                x, y = self.image_data_generator.random_transform(x, y, block=blockedaugmentations)

            x = image_utils.image_colour_convert(x, self.color_mode)

            x = crop_image(x, self.img_target_size, data_format=self.data_format)
            y = crop_image(y, self.gt_target_size, data_format=self.data_format)

            x = self.image_data_generator.standardize(x)

            if self.categoricaltarget:
                y = to_categorical(y, num_classes=self.nb_classes)

            batch_x[i] = x.astype(K.floatx())
            batch_y[i] = y.astype(K.floatx())

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                imgx = array_to_img(batch_x[i], self.data_format, scale=True)
                if self.categoricaltarget:
                    tmpy = numpy.expand_dims(numpy.argmax(batch_y[i], axis=-1), axis=len(y.shape))
                else:
                    tmpy = batch_y[i]
                imgy = array_to_img(tmpy * (255 / (self.nb_classes - 1)), self.data_format, scale=False)
                fname = os.path.splitext(os.path.basename(self.filenames[j]))[0]
                fname = '{prefix}_{origname}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                     origname=fname,
                                                                     hash=numpy.random.randint(1e4),
                                                                     format=self.save_format)
                imgx.save(os.path.join(self.save_to_dir, 'x_' + fname))
                imgy.save(os.path.join(self.save_to_dir, 'y_' + fname))

        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
