# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

import os
import numpy
import h5py
from keras.models import load_model
from keras.backend import epsilon
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import argparse
import keras.backend as K
from unet.losses import weighted_categorical_crossentropy
from utils import image_utils, config_utils, data_utils
import math


# Todo: should be modified to use an imagedatagenerator instead of loading the dataset into memory

def loadpatches(input_patch_path, stat_path, colour_mode, standardise_patches, normalise_patches, resample_factor, target_size, modellabel):

    from augmentation.live_augmentation import load_img
    from downsample_patches import __resampleimage

    # Read images from directory
    images_test = []
    #masks_test = []
    filenames = []
    for filename in os.listdir(os.path.join(input_patch_path)):
        image = image_utils.read_image(os.path.join(input_patch_path, filename)).astype(numpy.float32)

        img_size = image.shape[:-1]

        if resample_factor:
            image = __resampleimage(image, math.ceil(img_size[0] / resample_factor))
            img_size = image.shape[:-1]

        if img_size != target_size:

            if img_size[0] < target_size[0] or img_size[1] < target_size[1]:
                raise ValueError('Invalid cropped image size (%s). Image is %d x %d and target size is %d x %d.' % (
                    zos.path.join(input_patch_path, filename), img_size[0], img_size[1], target_size[0], target_size[1]))

            if (img_size[0] - target_size[0]) % 2 != 0:
                raise ValueError(
                    'Invalid cropped image size. There should be an even difference between the image and target heights')

            if (img_size[1] - target_size[1]) % 2 != 0:
                raise ValueError(
                    'Invalid cropped image size. There should be an even difference between the image and target widths')

            diffs = numpy.subtract(img_size, target_size)
            diffs //= 2
            image = image[diffs[0]:img_size[0] - diffs[0], diffs[1]:img_size[1] - diffs[1]]

        # image = load_img(os.path.join(input_patch_path, filename), grayscale=False, target_size=target_size).astype(numpy.float32)

        image = image_utils.image_colour_convert(image, colour_mode)
        images_test.append(image)

        #mask = image_utils.read_image(os.path.join(input_patch_path, 'gts', classname, filename)).astype(numpy.float32)
        #masks_test.append(mask)

        filenames.append(filename)

    images_test = numpy.array(images_test)
    #masks_test = numpy.array(masks_test)[..., None]

    #perm = numpy.random.permutation(images_test.shape[0])
    #numpy.take(images_test, perm, axis=0, out=images_test)
    #images_test = images_test[1:100, :, :]

    if standardise_patches:
        for idx, sample in enumerate(images_test):
            images_test[idx, ] = data_utils.standardise_sample(images_test[idx, ])

    if normalise_patches:
        # Read normalisation statistics
        statsfilename = os.path.join(stat_path, "models", "normalisation_stats." + modellabel + ".hdf5")
        with h5py.File(statsfilename, "r") as f:
            mean = f["stats"][0]
            stddev = f["stats"][1]

        # Normalise data
        for idx, sample in enumerate(images_test):
            images_test[idx, ] = data_utils.normalise_sample(images_test[idx, ], mean, stddev)

    return images_test, filenames

def getdataset(input_patch_path, stat_path, colour_mode, standardise_patches, normalise_patches, modellabel):

    # Read images from directory
    classnames = [x[1] for x in os.walk(os.path.join(input_patch_path, 'images'))][0]
    class_number = len(classnames)
    images_test = []
    masks_test = []
    filenames = []
    for classname in classnames:
        for filename in os.listdir(os.path.join(input_patch_path, 'images', classname)):

            image = image_utils.read_image(os.path.join(input_patch_path, 'images', classname, filename)).astype(numpy.float32)

            image = image_utils.image_colour_convert(image, colour_mode)

            images_test.append(image)

            mask = image_utils.read_image(os.path.join(input_patch_path, 'gts', classname, filename)).astype(numpy.float32)
            masks_test.append(mask)

            filenames.append(filename)

    images_test = numpy.array(images_test)
    masks_test = numpy.array(masks_test)[..., None]

    if standardise_patches:
        for idx, sample in enumerate(images_test):
            images_test[idx, ] = data_utils.standardise_sample(images_test[idx, ])

    if normalise_patches:
        # Read normalisation statistics
        statsfilename = os.path.join(stat_path, "models", "normalisation_stats." + modellabel + ".hdf5")
        with h5py.File(statsfilename, "r") as f:
            mean = f["stats"][0]
            stddev = f["stats"][1]

        # Normalise data
        for idx, sample in enumerate(images_test):
            images_test[idx, ] = data_utils.normalise_sample(images_test[idx, ], mean, stddev)

    return images_test, masks_test, class_number, filenames


def testunet(modellabel, images_test, masks_test, batch_size, output_path, filenames):

    # Cast data to floatx
    images_test = K.cast_to_floatx(images_test)
    if masks_test:
        masks_test = K.cast_to_floatx(masks_test)

    modelfilename = 'unet_best.' + modellabel + '.hdf5'

    UNet = load_model(os.path.join(output_path, 'models', modelfilename)) #, custom_objects={'loss': weighted_categorical_crossentropy()})

    class_number = int(UNet.outputs[0].shape[-1])

    if masks_test:
        acc = UNet.evaluate(x=images_test, y=masks_test, batch_size=batch_size)
        print(UNet.metrics_names)
        print(acc)

    y_pred = UNet.predict(images_test, batch_size=batch_size)

    img_output_path = os.path.join(output_path, 'predictions', modellabel)
    os.makedirs(img_output_path, exist_ok=True)

    for pred, filename in zip(y_pred, filenames):
        for c in range(class_number):
            image_utils.save_image((pred[..., c]*255).astype(numpy.uint8), os.path.join(img_output_path, os.path.basename(filename) + '_' + str(c) + '.png'))

    y_pred = numpy.argmax(y_pred, axis=-1)[..., None]

    if masks_test:
        print("Class accuracy:")
        for c in range(class_number):
            print("%d = %f" % (c, numpy.sum(numpy.logical_and(masks_test == c, y_pred == c)) / (numpy.sum(masks_test == c) + epsilon())))

        print("Overall accuracy:")
        print(accuracy_score(masks_test.flatten(), y_pred.flatten()))
        print(classification_report(masks_test.flatten(), y_pred.flatten()))
        print("Confusion Matrix:")
        c = confusion_matrix(masks_test.flatten(), y_pred.flatten())
        print(c)

        print("F1 score: %f" % f1_score(masks_test.flatten(), y_pred.flatten()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a UNet model.')

    parser.add_argument('label', type=str, help='the label of the model to be tested')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-d', '--directory', type=str, help='directory containing the test patches')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()
    config = config_utils.readconfig(os.path.join(config['detector.outputpath'], 'code.' + args.label + '.cfg'))

    patch_input_path = os.path.join(config['detector.inputpath'], 'test')

    if not args.directory:
        images_test, masks_test, class_number, filenames = getdataset(patch_input_path, config['detector.outputpath'],
                                                                      config['detector.colour_mode'],
                                                                      config['normalisation.standardise_patches'],
                                                                      config['normalisation.normalise_patches'],
                                                                      args.label)
    else:
        images_test, filenames = loadpatches(args.directory, config['detector.outputpath'],
                                             config['detector.colour_mode'],
                                             config['normalisation.standardise_patches'],
                                             config['normalisation.normalise_patches'],
                                             2,
                                             [256, 256],
                                             args.label)
        masks_test = None

    testunet(args.label, images_test, masks_test, config['segmentation.batch_size'], config['detector.outputpath'], filenames)
