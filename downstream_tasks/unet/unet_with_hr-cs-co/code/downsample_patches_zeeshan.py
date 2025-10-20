# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

import os
import warnings
import argparse
from utils import image_utils, config_utils
from skimage import transform
import numpy
import shutil
import json
import tensorflow as tf


def resamplepatches(patchpath, lod, target_patch_size):

    for f in os.listdir(patchpath):
        if f.endswith('.png'):
            print('resizing: {} | level-of-details: {} | target-size: {}'.format(f, lod, target_patch_size))
            new_fileName = f.rsplit('_', 1)[0]
            mask = image_utils.read_image(os.path.join(patchpath, f))

            #target_patch_size = numpy.divide(mask.shape, (4, 4))
            repetitions = (512 // target_patch_size) // 2
            print(repetitions)
            mask = __resamplegt(mask, repetitions)

            mask = numpy.rint(mask)

            new_patchpath = os.path.join(patchpath, 'smaller_size')
            os.makedirs(new_patchpath, exist_ok=True)

            outfilename = os.path.join(new_patchpath, '{}_lod{}.png'.format(new_fileName, lod))
            image_utils.save_image(mask.astype('uint8'), outfilename)
            shutil.copyfile(os.path.join(patchpath, 'class_labels.json'), os.path.join(new_patchpath, '{}_lod{}.json'.format(new_fileName, lod)))



def __resamplegt(image, repetitions):
    if True: #target_patch_size[1] < image.shape[1] and target_patch_size[0] < image.shape[0]:
        if len(image.shape) > 1:
            image = image_utils.downsample_gt_image(image[0:-1, 0:-1], repetitions)
        else:
            raise Exception("Can only downsample 1 channel ground truth masks")
    else:
        if target_patch_size > image.shape[1]:
            warnings.warn('Will only downsample patches. Extract patches with a higher resolution if necessary.')

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downsample GroundTruths.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    path = config['extraction.groundtruthpath']
    resamplepatches(path, config['general.lod'], config['detector.patch_size'])
