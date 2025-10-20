# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
import sys
import json
import datetime
import h5py
import shutil
import warnings
import glob

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from augmentation.live_augmentation import ImageDataGenerator
import tensorflow.keras.backend as K
import numpy
from unet import unet_models
from unet.callbacks import CheckpointTrainingPatches, ModelState, SaveHistory, ReduceLearningRate, DisplayLearningRateValue
from utils import image_utils, config_utils, data_utils, filepath_utils
from train_udagan import __save_training_history, getdataset
from tensorflow.keras.models import load_model
from utils.select_gpu import pick_gpu_lowest_memory


def resumetrainunet(number_of_classes, class_weights, epochs, output_path, label,
                    initialepoch=None, val_loss=None,config=None):
    """

    trainunet: Create and train a network based on the parameters contained within the configuration file
        - create a directory named 'graphs' in which the training and validation history are stored
        - create a directory named 'models' in which the trained network is saved


    :param number_of_classes: (int)  the number of classes used in the data
    :param class_weights: list of (int) the weight of each classes put in the list format (currently unused)
    :param epochs: (int) the number of epochs used in training
    :param output_path: (string) the path where the results are to be stored
    :param configfilename: (string) the path to the config file, it is copied into the output_path
    :return: (string) the label of the network
    """

    # Delete any results if network label already exists
    shutil.rmtree(os.path.join(config['detector.outputpath'], 'detections', label), ignore_errors=True)
    shutil.rmtree(os.path.join(config['detector.outputpath'], 'segmentations', label), ignore_errors=True)
    shutil.rmtree(os.path.join(config['detector.outputpath'], 'training_patches', label), ignore_errors=True)

    # Create output directories
    os.makedirs(os.path.join(config['detector.outputpath'], "models"), exist_ok=True)
    os.makedirs(os.path.join(config['detector.outputpath'], "graphs"), exist_ok=True)

    # Read configuration
    if config is None:
        raise RuntimeError('need to send configuration file')

    model_filename_latest = os.path.join(config['detector.outputpath'], "models",
                                         f"{config['detector.segmentationmodel']}_latest." + label + ".hdf5")
    use_best = False
    if os.path.exists(model_filename_latest):
        print('Loading latest model ....\n')
        
    else:
        print('There is no latest model!\n')
        print('Loading best model ...\n')
        use_best = True
        model_filename_latest = os.path.join(config['detector.outputpath'], "models",
                                             f"{config['detector.segmentationmodel']}_best." + label + ".hdf5")

    print(f"previously saved latest model: {model_filename_latest}")

    UNet = load_model(model_filename_latest)
    model_filename_latest = os.path.join(config['detector.outputpath'], "models",
                                         f"{config['detector.segmentationmodel']}_latest." + label + ".hdf5")

    inp_shape = UNet.inputs[0].get_shape().as_list()[1:3]
    otp_shape = UNet.outputs[0].get_shape().as_list()[1:3]

    if config['detector.weight_samples']:
        cattarget = True
    else:
        cattarget = False

    if 'stain' in config['augmentation.methods']:
        filePath = filepath_utils.FilepathGenerator(config)
    else:
        filePath = None

    if config['normalisation.normalise_patches']:
        statsfilename = os.path.join(output_path, 'models', "normalisation_stats." + label + ".hdf5")
        with h5py.File(statsfilename, "r") as f:
            mean = f["stats"][0]
            stddev = f["stats"][1]
    else:
        mean = None
        stddev = None

    augmentationclassblock = {'background': ['stain', 'stain_transfer', 'channel']}

    if config['augmentation.live_augmentation']:
        augmentationparameters = {}
        augmentationparameters['affine_rotation_range'] = config['augmentation.affine_rotation_range']
        augmentationparameters['affine_width_shift_range'] = config['augmentation.affine_width_shift_range']
        augmentationparameters['affine_height_shift_range'] = config['augmentation.affine_height_shift_range']
        augmentationparameters['affine_rescale'] = config['augmentation.affine_rescale']
        augmentationparameters['affine_zoom_range'] = config['augmentation.affine_zoom_range']
        augmentationparameters['affine_horizontal_flip'] = config['augmentation.affine_horizontal_flip']
        augmentationparameters['affine_vertical_flip'] = config['augmentation.affine_vertical_flip']
        augmentationparameters['elastic_sigma'] = config['augmentation.elastic_sigma']
        augmentationparameters['elastic_alpha'] = config['augmentation.elastic_alpha']
        augmentationparameters['smotenneighbours'] = config['augmentation.smotenneighbours']
        augmentationparameters['stain_alpha_range'] = config['augmentation.stain_alpha_range']
        augmentationparameters['stain_beta_range'] = config['augmentation.stain_beta_range']
        augmentationparameters['blur_sigma_range'] = config['augmentation.blur_sigma_range']
        augmentationparameters['noise_sigma_range'] = config['augmentation.noise_sigma_range']
        augmentationparameters['bright_factor_range'] = config['augmentation.bright_factor_range']
        augmentationparameters['contrast_factor_range'] = config['augmentation.contrast_factor_range']
        augmentationparameters['colour_factor_range'] = config['augmentation.colour_factor_range']
        augmentationparameters['colour_transfer_targets'] = config['trainingstrategy.targetstainings']
        augmentationparameters['gan_model_path'] = os.path.join(config['trainingstrategy.gan_model_path'],
                                                                config['general.staincode'])
        augmentationparameters['saved_translations_path'] = config['trainingstrategy.saved_translations_path']
        augmentationparameters['model_suffixes'] = config['trainingstrategy.model_suffixes']
        augmentationparameters['transfer_model'] = config['trainingstrategy.transfer_model']
        augmentationparameters['transfer_choice'] = config['trainingstrategy.transfer_choice']

        print(f"ImageDataGenerator with {config['augmentation.methods']} augmentations...")
        train_gen = ImageDataGenerator(methods=config['augmentation.methods'],
                                       augmentationparameters=augmentationparameters,
                                       fill_mode='reflect',
                                       standardise_sample=config['normalisation.standardise_patches'],
                                       samplewise_normalise=config['normalisation.normalise_patches'],
                                       nb_classes=number_of_classes,
                                       categoricaltarget=cattarget)
    else:
        train_gen = ImageDataGenerator(standardise_sample=config['normalisation.standardise_patches'],
                                       samplewise_normalise=config['normalisation.normalise_patches'],
                                       nb_classes=number_of_classes,
                                       categoricaltarget=cattarget)

    train_flow = train_gen.flow_from_directory(os.path.join(config['detector.traininputpath'], 'train'),
                                               filepath=filePath,
                                               img_target_size=(inp_shape[0], inp_shape[1]),
                                               gt_target_size=(otp_shape[0], otp_shape[1]),
                                               color_mode=config['detector.colour_mode'],
                                               batch_size=config['detector.batch_size'],
                                               shuffle=True,
                                               dataset_mean=mean,
                                               dataset_std=stddev,
                                               augmentationclassblock=augmentationclassblock)

    validationaugmentation = []
    validationaugmentationparameters = {}
    if 'channel' in config['augmentation.methods']:
        validationaugmentation = ['channel']
    if 'stain_transfer' in config['augmentation.methods']:
        validationaugmentation = ['stain_transfer']
        validationaugmentationparameters['colour_transfer_targets'] = config['trainingstrategy.targetstainings']
        validationaugmentationparameters['saved_translations_path'] = config['trainingstrategy.saved_translations_path']
        validationaugmentationparameters['gan_model_path'] = os.path.join(config['trainingstrategy.gan_model_path'],
                                                                          config['general.staincode'])
        validationaugmentationparameters['model_suffixes'] = config['trainingstrategy.model_suffixes']
        validationaugmentationparameters['transfer_model'] = config['trainingstrategy.transfer_model']
        validationaugmentationparameters['transfer_choice'] = config['trainingstrategy.transfer_choice']

    validation_gen = ImageDataGenerator(methods=validationaugmentation,
                                        augmentationparameters=validationaugmentationparameters,
                                        standardise_sample=config['normalisation.standardise_patches'],
                                        samplewise_normalise=config['normalisation.normalise_patches'],
                                        nb_classes=number_of_classes,
                                        categoricaltarget=cattarget)

    valid_flow = validation_gen.flow_from_directory(os.path.join(config['detector.validationinputpath'], 'validation'),
                                                    img_target_size=(inp_shape[0], inp_shape[1]),
                                                    gt_target_size=(otp_shape[0], otp_shape[1]),
                                                    color_mode=config['detector.colour_mode'],
                                                    batch_size=config['detector.batch_size'],
                                                    shuffle=True,
                                                    dataset_mean=mean,
                                                    dataset_std=stddev,
                                                    augmentationclassblock=augmentationclassblock)

    callbacklist = []
    model_filename = os.path.join(config['detector.outputpath'], "models",
                                  f"{config['detector.segmentationmodel']}_best." + label + ".hdf5")

    callbacklist.append(ModelCheckpoint(model_filename_latest, verbose=1, save_freq=int(1 * len(train_flow))))

    historyfile = os.path.join(output_path, "models", f"{config['detector.segmentationmodel']}_history." + label + ".json")
    model_history = SaveHistory(historyfile, read_existing=True)
    callbacklist.append(model_history)

    if not initialepoch:
        if 'val_loss' in model_history.history:
            val_loss_list = model_history.history['val_loss']
            if (len(val_loss_list) > 60) and use_best:
                val_loss_list = val_loss_list[:60]
                
            if use_best:
                initialepoch = val_loss_list.index(min(val_loss_list)) + 1
            else:
                initialepoch = len(model_history.history['val_loss'])
            print(f'Training is resumed at {initialepoch} epoch')
        else:
            raise ValueError(f'Number of epochs could not be read from {historyfile}')

    model_checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    if not val_loss:
        if 'val_loss' in model_history.history:
            val_loss_list = model_history.history['val_loss']
            if (len(val_loss_list) > 60) and (use_best==True):
                val_loss_list = val_loss_list[:60]
            model_checkpoint.best = min(val_loss_list)
        else:
            raise ValueError(f'Best loss value could not be read from {historyfile}')
    else:
        model_checkpoint.best = val_loss
    print(f'Best validation loss is:{model_checkpoint.best}')
    callbacklist.append(model_checkpoint)

    if config['detector.reducelr']:
        raise ValueError("Reduce learning rate cannot be used when resuming training")
        #callbacklist.append(
        #    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001,
        #                      cooldown=0, min_lr=0))

    if config['detector.earlyStopping']:
        callbacklist.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'))

    print("Number of classes: %s" % number_of_classes)
    print("Patch size: %s" % str(inp_shape))

    UNet.fit(train_flow, initial_epoch=initialepoch, epochs=epochs, shuffle=True,
             callbacks=callbacklist, validation_data=valid_flow, use_multiprocessing=True, workers=8, max_queue_size=650)

    __save_training_history(os.path.join(output_path, "graphs"), historyfile, label)

    print("Written model with label: %s" % label)

    os.remove(model_filename_latest)

    return label


def pretrained_model_path(conf):
    return os.path.join(conf['general.homepath'], 'saved_models/SSL/Nephrectomy',
                        conf['transferlearning.pretrained_ssl_model'], conf['detector.segmentationmodel'],
                        conf['transferlearning.pretrained_ssl_model_rep'],
                        conf['transferlearning.pretrained_ssl_model_name'], 'models',
                        conf['transferlearning.pretrained_ssl_model_at_epoch'])


def pretrained_model_stats_file(conf):
    return os.path.join(conf['general.homepath'], 'saved_models/SSL/Nephrectomy',
                        conf['transferlearning.pretrained_ssl_model'], 'data_statistics/normalisation_stats.hdf5')


def derived_parameters(conf, arguments):
    conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model
    conf['transferlearning.pretrained_ssl_model_rep'] = arguments.pretrained_ssl_model_rep
    conf['transferlearning.pretrained_ssl_model_name'] = arguments.pretrained_ssl_model_name
    conf['transferlearning.pretrained_ssl_model_at_epoch'] = arguments.pretrained_ssl_model_at_epoch
    if conf['transferlearning.finetune']:
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable
        conf['transferlearning.pretrained_ssl_model_path'] = pretrained_model_path(conf)
        conf['transferlearning.pretrained_ssl_model_stats_file'] = pretrained_model_stats_file(conf)
    else:
        conf['transferlearning.pretrained_ssl_model_path'] = None
        conf['transferlearning.pretrained_ssl_model_stats_file'] = None

    conf['detector.patchstrategy'] = arguments.patch_strategy
    conf['detector.percentN'] = arguments.percent_N

    if conf['detector.percentN'] == "percent_100":
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'patches')
    else:
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'separated_patches',
                                                       conf['detector.patchstrategy'], conf['detector.percentN'])

    conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'])

    if conf['transferlearning.finetune']:
        if conf['transferlearning.pretrained_model_trainable']:
            conf['detector.reduce_learning_rate_epoch'] = arguments.reduce_learning_rate_epoch

    if conf['transferlearning.finetune']:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'],
                                                   conf['transferlearning.pretrained_ssl_model'],
                                                   conf['transferlearning.pretrained_ssl_model_name'])
    else:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'])

    conf['segmentation.segmentationpath'] = os.path.join(conf['detector.outputpath'],
                                                         conf['segmentation.segmentationpath'])
    conf['segmentation.detectionpath'] = os.path.join(conf['detector.outputpath'],
                                                      conf['segmentation.detectionpath'])
    return conf


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-l', '--label', type=str, help='the label of the model to resume training')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
    parser.add_argument('-e', '--epoch', type=int, help='epoch to resume training (0 indexed)')
    parser.add_argument('-v', '--valloss', type=float, help='best valloss found so far')

    # Adding parameters to finetune the UNet with pretrained Self Supervised Learning Models (SimCLR, Byol, CSCO, etc)
    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, default='SimCLR')
    parser.add_argument('-pmn', '--pretrained_ssl_model_name', type=str, default='Base_Scale')
    parser.add_argument('-pme', '--pretrained_ssl_model_at_epoch', type=str, default='model_epoch199.h5')
    parser.add_argument('-pmr', '--pretrained_ssl_model_rep', type=str, default='rep1')

    # Adding parameters to specify the data strategy
    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pn', '--percent_N', type=str, default='percent_1')
    parser.add_argument('-pmt', '--pretrained_model_trainable', default=False,
                        type=lambda x: str(x).lower() in ['true', '1', 'yes'])
    parser.add_argument('-rle', '--reduce_learning_rate_epoch', type=str, default="None")

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())

    print("Selected GPU : " + os.environ["CUDA_VISIBLE_DEVICES"])

    print("\n")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
       raise ValueError("Need to send configuration file")

    if args.reduce_learning_rate_epoch != 'None':
        args.reduce_learning_rate_epoch = int(args.reduce_learning_rate_epoch)

    patch_input_path = os.path.join(config['detector.traininputpath'], 'train')

    if args.label and not filepath_utils.validlabel(args.label):
        raise ValueError('The label should not contain periods "."')

    config = derived_parameters(config, arguments=args)

    number_of_classes = len(config['extraction.class_definitions'])
    class_weights = [1] * number_of_classes

    resumetrainunet(number_of_classes, class_weights, config['detector.epochs'], config['detector.outputpath'],
                    initialepoch=args.epoch, label=args.label, val_loss=args.valloss, config=config)

