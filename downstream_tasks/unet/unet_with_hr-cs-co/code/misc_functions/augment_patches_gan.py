# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

from augmentation import smote, elastic_transform, affine_transform, stain_transfer_transform, blur_transform, noise_transform, enhance_transform
import glob
import os
import argparse
import shutil
from utils import config_utils, filepath_utils
import random
from cycgan import CycleGAN_models


def augmentpatches(data_path, outpath, classnames, samples_per_class, filePath=None):
    """
    augmentpatches:

    :param data_path:
    :param outpath:
    :param classnames:
    :param samples_per_class:
    :param augmentationparameters:
    :param methods:
    :return:
    """


    transform_model = CycleGAN_models.CycleGAN()
    transform_model.combined.load_weights('/home/lampert/transfer_gans/all_to_02/25_106_107_108/combined.h5')
    transfer_fun = {}
    transfer_fun['25_106_107_108'] = transform_model.g_AB

    for classname, nsamples in zip(classnames, samples_per_class):

        ##
        # Pick random transform from methods
        ##

        # methodindexes = [random.randint(0, nmethods - 1) for x in range(0, classname_samples[1])]
        #
        # for i in range(0, nmethods):
        #
        #     nsamples = methodindexes.count(i)
        #
        #     if nsamples > 0:
        #         if methods[i] == 'affine':
        #             affine_transform.generate_from_directory(classname_samples[0], nsamples, data_path, outpath,
        #                                            rotation_range=augmentationparameters['affine_rotation_range'],
        #                                            width_shift_range=augmentationparameters['affine_width_shift_range'],
        #                                            height_shift_range=augmentationparameters['affine_height_shift_range'],
        #                                            rescale=augmentationparameters['affine_rescale'],
        #                                            zoom_range=augmentationparameters['affine_zoom_range'],
        #                                            horizontal_flip=augmentationparameters['affine_horizontal_flip'],
        #                                            vertical_flip=augmentationparameters['affine_vertical_flip'],
        #                                            fill_mode='reflect',
        #                                            cval=0.)
        #
        #         if methods[i] == 'smote':
        #             smote.generate_from_directory(classname_samples[0], nsamples, data_path, outpath,
        #                                           augmentationparameters['smotenneighbours'])
        #
        #         if methods[i] == 'elastic':
        #             elastic_transform.generate_from_directory(classname_samples[0], nsamples, data_path, outpath,
        #                                                       sigma=augmentationparameters['elastic_sigma'],
        #                                                       alpha=augmentationparameters['elastic_alpha'])
        #
        #         if methods[i] == 'stain':
        #             stain_transform.generate_from_directory(classname_samples[0], nsamples, data_path, outpath, staincode)
        #
        #         if methods[i] == 'blur':
        #             blue_transform.generate_from_directory(classname_samples[0], nsamples, data_path, outpath,
        #                                                    sigma=augmentationparameters['blur_sigma_range'])
        #
        #         if methods[i] == 'noise':
        #             noise_transform.generate_from_directory(classname_samples[0], nsamples, data_path, outpath,
        #                                                     sigma=augmentationparameters['noise_sigma_range'])

        ##
        # Apply all transforms
        ##

        data_path_orig = data_path

        stain_transfer_transform.generate_from_directory(classname, nsamples, data_path, outpath, '', transfer_fun,
                                                    changefilename=False)
        data_path = outpath

        data_path = data_path_orig


def balanceclasses(data_path, outpath, filePath=None):

    max_classnumer = -1

    classnames = [x[1] for x in os.walk(os.path.join(data_path, 'images'))][0]

    counts = {}
    for classname in classnames:
        counts[classname] = len([f for f in os.listdir(os.path.join(data_path, 'images', classname))
                                 if os.path.isfile(os.path.join(data_path, 'images', classname, f))])
        if counts[classname] > max_classnumer:
            max_classnumer = counts[classname]

    print('Found %s samples' % str(counts))

    samples_per_class = []
    for classname in classnames:
        samples_per_class.append(max_classnumer - counts[classname])

    print('Generating %s samples of classes %s' % (str(samples_per_class), str(classnames)))

    augmentpatches(data_path, outpath, classnames, samples_per_class, filePath=filePath)


def multiplysamplenumbers(data_path, outpath, factor, filePath=None):

    classnames = [x[1] for x in os.walk(os.path.join(data_path, 'images'))][0]

    counts = {}
    for classname in classnames:
        counts[classname] = len([f for f in os.listdir(os.path.join(data_path, 'images', classname))
                                 if os.path.isfile(os.path.join(data_path, 'images', classname, f))])

    print('Found %s samples' % str(counts))

    samples_per_class = []
    for classname in classnames:
        samples_per_class.append(counts[classname] * factor)

    print('Generating %s samples of classes %s' % (str(samples_per_class), str(classnames)))

    augmentpatches(data_path, outpath, classnames, samples_per_class, filePath=filePath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
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

#    if not config['augmentation.use_augmentation']:
#        print("Augmentation not activated in the config file")
#        raise SystemExit(0)
#    elif config['augmentation.live_augmentation']:
#        print("Live augmentation activated in the config file")
#        raise SystemExit(0)

#    if os.path.isdir(config['augmentation.patchpath']):
#        shutil.rmtree(config['augmentation.patchpath'])

#    if 'stain' in config['augmentation.methods']:
#        filePath = filepath_utils.FilepathGenerator(config)
#    else:
    filePath = None

#    for t in ['train']:
    for t in ['validation']:

        if config['augmentation.balanceclasses']:
            balanceclasses(os.path.join(config['extraction.patchpath'], t), os.path.join(config['augmentation.patchpath'], t), filePath=filePath)

        if config['augmentation.multiplyexamples']:
            multiplysamplenumbers(os.path.join(config['extraction.patchpath'], t), os.path.join(config['augmentation.patchpath'], t), config['augmentation.multiplyfactor'], filePath=filePath)

            shutil.copyfile(os.path.join(config['extraction.patchpath'], t, 'class_labels.json'),
                            os.path.join(config['augmentation.patchpath'], t, 'class_labels.json'))
