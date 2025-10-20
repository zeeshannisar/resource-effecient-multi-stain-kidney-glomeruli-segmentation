# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

import os
import warnings
import argparse
from utils import image_utils, config_utils
from skimage import transform
import numpy
import shutil


def resamplepatches(patchpath, processed_patch_path, target_patch_size):
    classnames = [x[1] for x in os.walk(os.path.join(patchpath, 'images'))][0]
    for classname in classnames:

        shutil.rmtree(os.path.join(processed_patch_path, 'images', classname), ignore_errors=True)
        shutil.rmtree(os.path.join(processed_patch_path, 'gts', classname), ignore_errors=True)
        os.makedirs(os.path.join(processed_patch_path, 'images', classname), exist_ok=True)
        os.makedirs(os.path.join(processed_patch_path, 'gts', classname), exist_ok=True)

        for f in os.listdir(os.path.join(patchpath, 'images', classname)):
            image = image_utils.read_image(os.path.join(patchpath, 'images', classname, f))
            image = __resampleimage(image, target_patch_size)

            outfilename = os.path.join(processed_patch_path, 'images', classname, os.path.basename(f))
            image_utils.save_image(image.astype('uint8'), outfilename)

        for f in os.listdir(os.path.join(patchpath, 'gts', classname)):
            mask = image_utils.read_image(os.path.join(patchpath, 'gts', classname, f))
            mask = __resamplegt(mask, target_patch_size)

            mask = numpy.rint(mask)

            outfilename = os.path.join(processed_patch_path, 'gts', classname,  os.path.basename(f))
            image_utils.save_image(mask.astype('uint8'), outfilename)


def __resampleimage(image, target_patch_size):
    if target_patch_size < image.shape[1]:
        if len(image.shape) == 3:
            image = transform.resize(image, (target_patch_size, target_patch_size, image.shape[2]), order=1, mode='reflect', clip=True, preserve_range=True, anti_aliasing=False)
        elif len(image.shape) == 2:
            image = transform.resize(image, (target_patch_size, target_patch_size), order=1, mode='reflect', clip=True, preserve_range=True, anti_aliasing=False)
        else:
            raise ValueError("Image should have 2 or 3 dimensions")
    else:
        if target_patch_size > image.shape[1]:
            warnings.warn('Will only downsample patches. Extract patches with a higher resolution if necessary.')

    return image


def __resamplegt(image, target_patch_size):
    if target_patch_size < image.shape[1]:
        if len(image.shape) > 1:
            image = image_utils.downsample_gt(image, target_patch_size)
        else:
            raise Exception("Can only downsample 1 channel ground truth masks")
    else:
        if target_patch_size > image.shape[1]:
            warnings.warn('Will only downsample patches. Extract patches with a higher resolution if necessary.')

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    paths = [config['extraction.patchpath']]
    if config['augmentation.use_augmentation'] and (not config['augmentation.live_augmentation']):
        paths.append(config['augmentation.patchpath'])

        # Check that augmented and original patch classes match
        with open(os.path.join(config['extraction.patchpath'], 'train', 'class_labels.json'), 'r') as fp:
            patchgtlabels = json.load(fp)
        with open(os.path.join(config['augmentation.patchpath'], 'train', 'class_labels.json'), 'r') as fp:
            augpatchgtlabels = json.load(fp)
        if not patchgtlabels == augpatchgtlabels:
            raise ValueError("Patch and augmented patch gt labels differ")

    shutil.rmtree(os.path.join(config['detector.inputpath']), ignore_errors=True)

    # We've done processing so far on maximum resolution, so resample to desired resolution
    for p in paths:
        for t in ['train', 'validation', 'test']:
            if os.path.isdir(os.path.join(p, t)):
                resamplepatches(os.path.join(p, t), os.path.join(config['detector.inputpath'], t), config['detector.patch_size'])
                shutil.copyfile(os.path.join(p, t, 'class_labels.json'), os.path.join(config['detector.inputpath'], t, 'class_labels.json'))
