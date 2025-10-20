# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

import matplotlib
matplotlib.use('Agg')
import os
import argparse
import sys
import numpy
import matplotlib.pyplot as plt
import json
import datetime
import h5py
import shutil
import warnings
import glob

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from unet import unet_models
from train_unet import __save_training_history
from unet.callbacks import CheckpointTrainingPatches, SaveHistory, ReduceLearningRate, DisplayLearningRateValue
from utils import image_utils, config_utils, data_utils, filepath_utils
from augmentation.live_augmentation import ImageDataGenerator


def resumetrainunet(number_of_classes, class_weights, epochs, output_path, label, initialepoch=None, val_loss=None):
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
    # shutil.rmtree(os.path.join(output_path, 'detections', label), ignore_errors=True)
    # shutil.rmtree(os.path.join(output_path, 'segmentations', label), ignore_errors=True)
    # shutil.rmtree(os.path.join(output_path, 'training_patches', label), ignore_errors=True)

    # Create output directories
    os.makedirs(os.path.join(output_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "graphs"), exist_ok=True)

    # Read configuration
    # config = config_utils.readconfig(os.path.join(output_path, "sysmifta." + label + ".cfg"))

    # Read configuration
    # with open(os.path.join(output_path, 'SSL.' + label + '.json'), 'r') as f:
    #     config = json.load(f)


    model_filename_latest = os.path.join(output_path, "models", f"{config['detector.segmentationmodel']}_latest." + label + ".keras")

    UNet = load_model(model_filename_latest)

    # inp_shape = UNet.inputs[0].get_shape().as_list()[1:3]
    # otp_shape = UNet.outputs[0].get_shape().as_list()[1:3]
    inp_shape = list(UNet.inputs[0].shape)[1:3]
    otp_shape = list(UNet.outputs[0].shape)[1:3]

    filePath = filepath_utils.FilepathGenerator(config)

    if config['normalisation.normalise_patches']:
        statsfilename = os.path.join(output_path, 'models', "normalisation_stats." + label + ".hdf5")
        with h5py.File(statsfilename, "r") as f:
            mean = f["stats"][0]
            stddev = f["stats"][1]
    else:
        mean = None
        stddev = None

    augmentationclassblock = {'background': ['stain', 'stain_transfer', 'channel']}

    if config['detector.transferlearning']:
        pretrained_model = config['transferlearning.pretrained_ssl_model']
    else:
        pretrained_model = None

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
        augmentationparameters['colour_transfer_staindatadir'] = config['general.datapath']
        augmentationparameters['colour_transfer_targets'] = config['trainingstrategy.targetstainings']
        augmentationparameters['colour_transfer_staindatadir'] = config['general.datapath']

        print(f"ImageDataGenerator with {config['augmentation.methods']} augmentations...")
        train_gen = ImageDataGenerator(methods=config['augmentation.methods'],
                                       augmentationparameters=augmentationparameters,
                                       fill_mode='reflect',
                                       standardise_sample=config['normalisation.standardise_patches'],
                                       samplewise_normalise=config['normalisation.normalise_patches'],
                                       nb_classes=number_of_classes,
                                       categoricaltarget=False)
    else:
        train_gen = ImageDataGenerator(standardise_sample=config['normalisation.standardise_patches'],
                                       samplewise_normalise=config['normalisation.normalise_patches'],
                                       nb_classes=number_of_classes,
                                       categoricaltarget=False)

    train_flow = train_gen.flow_from_directory(os.path.join(config['detector.traininputpath'], 'train'),
                                               filepath=filePath,
                                               img_target_size=(inp_shape[0], inp_shape[1]),
                                               gt_target_size=(otp_shape[0], otp_shape[1]),
                                               color_mode=config['detector.colour_mode'],
                                               batch_size=config['detector.batch_size'],
                                               shuffle=True,
                                               dataset_mean=mean,
                                               dataset_std=stddev,
                                               augmentationclassblock=augmentationclassblock,
                                               pretrained_models=pretrained_model)

    validationaugmentation = []
    validationaugmentationparameters = {}
    if 'channel' in config['augmentation.methods']:
        validationaugmentation = ['channel']
    if 'stain_transfer' in config['augmentation.methods']:
        validationaugmentation = ['stain_transfer']
        validationaugmentationparameters['colour_transfer_targets'] = config['trainingstrategy.targetstainings']
        validationaugmentationparameters['colour_transfer_staindatadir'] = config['general.datapath']

    validation_gen = ImageDataGenerator(methods=validationaugmentation,
                                        augmentationparameters=validationaugmentationparameters,
                                        standardise_sample=config['normalisation.standardise_patches'],
                                        samplewise_normalise=config['normalisation.normalise_patches'],
                                        nb_classes=number_of_classes,
                                        categoricaltarget=False)

    valid_flow = validation_gen.flow_from_directory(os.path.join(config['detector.validationinputpath'], 'validation'),
                                                    filepath=filePath,
                                                    img_target_size=(inp_shape[0], inp_shape[1]),
                                                    gt_target_size=(otp_shape[0], otp_shape[1]),
                                                    color_mode=config['detector.colour_mode'],
                                                    batch_size=config['detector.batch_size'],
                                                    shuffle=True,
                                                    dataset_mean=mean,
                                                    dataset_std=stddev,
                                                    augmentationclassblock=augmentationclassblock,
                                                    pretrained_models=pretrained_model)

    callbacklist = []
    model_filename = os.path.join(output_path, "models", f"{config['detector.segmentationmodel']}_best." + label + ".keras")
    callbacklist.append(ModelCheckpoint(model_filename_latest, verbose=1, save_freq='epoch'))
    historyfile = os.path.join(output_path, "models", f"{config['detector.segmentationmodel']}_history." + label + ".json")
    model_history = SaveHistory(historyfile, read_existing=True)
    callbacklist.append(model_history)

    if not initialepoch:
        if 'val_loss' in model_history.history:
            initialepoch = len(model_history.history['val_loss'])
        else:
            raise ValueError('Number of epochs could not be read from %s' % historyfile)

    print("Training is resumed at epoch:", initialepoch)

    if config['detector.validationinputpath'] is not None:
        model_checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    else:
        model_checkpoint = ModelCheckpoint(model_filename, monitor='loss', verbose=1, save_best_only=True, mode='auto')

    if not val_loss:
        if 'val_loss' in model_history.history:
            model_checkpoint.best = min(model_history.history['val_loss'])
        else:
            raise ValueError('Best loss value could not be read from %s' % historyfile)
    else:
        model_checkpoint.best = val_loss

    print('Best loss is:', model_checkpoint.best)
    callbacklist.append(model_checkpoint)


    if config['detector.reducelr']:
        if initialepoch >= config['detector.reduceLR_patience'] and K.get_value(UNet.optimizer.learning_rate) > 0.00001:
            print("Reducing learning rate with a factor of 0.1 after %d epochs." % config['detector.reduceLR_patience'])
            UNet.optimizer.learning_rate.assign(K.get_value(UNet.optimizer.learning_rate) * 0.1)
        else:
            callbacklist.append(ReduceLearningRate(patience=config['detector.reduceLR_patience'], factor=0.1, verbose=1))

    print("Current learning rate: {:.6f}".format(float(K.get_value(UNet.optimizer.learning_rate))))
    
    if config['detector.earlyStopping']:
        callbacklist.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'))

    callbacklist.append(DisplayLearningRateValue())

    if config['detector.reducelr']:
        print(f"LR: {config['detector.learn_rate']} will be reduced with a factor of 0.1 at "
              f"{config['detector.reduceLR_percentile']}-th percentile ({config['detector.reduceLR_patience']}-th "
              f"epoch) of training.")
    else:
        print(f"LR: {config['detector.learn_rate']}")

    print("Number of classes: %s" % number_of_classes)
    print("Patch size: %s" % str(inp_shape))

    if config['detector.validationinputpath'] is not None:
        UNet.fit(train_flow, initial_epoch=initialepoch, epochs=(epochs - 100), shuffle=True, callbacks=callbacklist,
                 validation_data=valid_flow)#, verbose=2, use_multiprocessing=True, workers=12, max_queue_size=650)
    else:
        UNet.fit(train_flow, initial_epoch=initialepoch, epochs=(epochs - 100), shuffle=True, callbacks=callbacklist)
                 #use_multiprocessing=True)#, verbose=2, workers=8, max_queue_size=650)

    __save_training_history(os.path.join(output_path, "graphs"), historyfile, label, config['detector.validation_data'])

    print("Written model with label: %s" % label)

    if os.path.exists(model_filename_latest):
        os.remove(model_filename_latest)

    return label


def pretrained_csco_model_path(conf):
    return os.path.join(conf['general.workpath'], conf['general.additionalpath'], 'saved_models/postdoc',
                        'improve_kidney_glomeruli_segmentation/sysmifta/pretraining/final_selected_models',
                        f'hrcsco/csco_unet_encoder_{conf["general.staincode"]}.hdf5')


def derived_parameters(conf, arguments):
    conf['detector.patchstrategy'] = arguments.patch_strategy
    conf['detector.percentN'] = arguments.percent_N

    if arguments.validation_data.lower() == "none":
        config['detector.validation_data'] = None
    else:
        config['detector.validation_data'] = arguments.validation_data

    if conf['detector.percentN'] == "percent_100":
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'supervised_patches')
    else:
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'separated_supervised_patches',
                                                       conf['detector.patchstrategy'], conf['detector.percentN'])

    if config['detector.validation_data'] is None:
        conf['detector.validationinputpath'] = None

    if conf['detector.validation_data'].lower() == "respective_splits":
        if conf['detector.percentN'] == "percent_100":
            conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'],
                                                                'supervised_patches')
        else:
            conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'],
                                                                'separated_supervised_patches',
                                                                conf['detector.patchstrategy'],
                                                                conf['detector.percentN'])

    if conf['detector.validation_data'].lower() == "full":
        conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'], 'supervised_patches')

    if conf['detector.validation_data'] is None:
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], f'epochs_{conf["detector.epochs"]}',
                                                  'downstream_tasks_evaluation',
                                                  conf['detector.patchstrategy'], conf['detector.percentN'],
                                                  conf['general.staincode'], 'without_validation_data')

    if conf['detector.validation_data'].lower() == "respective_splits":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], f'epochs_{conf["detector.epochs"]}',
                                                  'downstream_tasks_evaluation',
                                                  conf['detector.patchstrategy'], conf['detector.percentN'],
                                                  conf['general.staincode'], 'respective_splits_validation_data')

    if conf['detector.validation_data'].lower() == "full":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], f'epochs_{conf["detector.epochs"]}',
                                                  'downstream_tasks_evaluation',
                                                  conf['detector.patchstrategy'], conf['detector.percentN'],
                                                  conf['general.staincode'], 'full_validation_data')

    if conf['detector.transferlearning']:
        conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model.lower()
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable

        if 'hrcsco' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.csco_model_path'] = pretrained_csco_model_path(conf)
        else:
            raise ValueError("Self-supervised learning based pretrained-models should be 'hrcsco'")

        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'],
                                                   conf['transferlearning.pretrained_ssl_model'])
    else:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'])

    conf['segmentation.segmentationpath'] = os.path.join(conf['detector.outputpath'],
                                                         conf['segmentation.segmentationpath'])
    conf['segmentation.detectionpath'] = os.path.join(conf['detector.outputpath'], conf['segmentation.detectionpath'])
    return conf


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Resume training unet.')

    parser.add_argument('-l', '--label', type=str, help='the label of the model to resume training')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
    parser.add_argument('-e', '--epoch', type=int, help='epoch to resume training (0 indexed)')
    parser.add_argument('-v', '--valloss', type=float, help='best valloss found so far')

    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, default='hrcsco')
    parser.add_argument('-pmt', '--pretrained_model_trainable', default=True,
                        type=lambda x: str(x).lower() in ['true', '1', 'yes'])

    # Adding parameters to specify the data strategy
    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pn', '--percent_N', type=str, default='percent_1')

    parser.add_argument('-lr', '--LR', type=str, default="0.0001")
    parser.add_argument('-lrd', '--LR_weightdecay', type=str, default="None")
    parser.add_argument('-rlrp', '--reduceLR_percentile', type=str, default='90')

    parser.add_argument('-vd', '--validation_data', type=str, default='full', help="none | respective_splits | full")
    parser.add_argument('-nte', '--num_training_epochs', type=int, default=250)

    start = datetime.datetime.now()
    args = parser.parse_args()


    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    patch_input_path = os.path.join(config['detector.traininputpath'], 'train')

    if args.label and not filepath_utils.validlabel(args.label):
        raise ValueError('The label should not contain periods "."')

    if args.num_training_epochs:
        config['detector.epochs'] = args.num_training_epochs
    
    config = derived_parameters(config, arguments=args)

    if args.LR != 'None':
        config['detector.learn_rate'] = float(args.LR)

    if args.LR_weightdecay != 'None':
        config['detector.LR_weightdecay'] = float(args.LR_weightdecay)
    else:
        config['detector.LR_weightdecay'] = None

    if args.reduceLR_percentile != 'None' and config['detector.transferlearning']:
        if config['detector.reducelr'] and config['transferlearning.pretrained_model_trainable']:
            config['detector.reduceLR_percentile'] = int(args.reduceLR_percentile)
            config['detector.reduceLR_patience'] = int((config['detector.reduceLR_percentile'] / 100) * (config['detector.epochs'] - 100))

    number_of_classes = len(config['extraction.class_definitions'])
    class_weights = [1] * number_of_classes

    print("\nConfiguration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))

    resumetrainunet(number_of_classes, class_weights, config['detector.epochs'], config['detector.outputpath'], 
                    initialepoch=args.epoch, label=args.label, val_loss=args.valloss)

