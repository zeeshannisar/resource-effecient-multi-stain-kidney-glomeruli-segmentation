import matplotlib
matplotlib.use('Agg')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from augmentation.live_augmentation import ImageDataGenerator
import tensorflow.keras.backend as K
import numpy
from unet.unet_models import SimCLR, BYOL, unet_decoder_above_SSL, baseline_unet_model
from unet.callbacks import SaveHistory, ReduceLearningRate, DisplayLearningRateValue
from utils import config_utils, filepath_utils
import matplotlib.pyplot as plt
import json
import datetime
import h5py
import shutil
from delete_model import delete_model
from unet.losses import weighted_categorical_crossentropy
from utils.image_utils import check_gt_validity
from utils.select_gpu import pick_gpu_lowest_memory


def __save_training_history(logpath, historyfile, label, validationinputpath):
    """
        Save the history of the model
    """

    with open(historyfile, 'r') as f:
        history = json.load(f)

    if 'lr' in history:
        history['lr'] = [float(v) for v in history['lr']]

    # Summarise loss history
    plt.clf()
    plt.plot(history['loss'])
    if validationinputpath is not None:
        plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if validationinputpath is not None:
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
        
    plt.gcf().savefig(os.path.join(logpath, 'loss_history.' + label + '.png'))
    # plt.show()

    # Summarise accuracy history
    plt.clf()
    plt.plot(history['sparse_categorical_accuracy'])
    if validationinputpath is not None:
        plt.plot(history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if validationinputpath is not None:
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
        
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, 1))
    plt.gcf().savefig(os.path.join(logpath, 'acc_history.' + label + '.png'))
    # plt.show()

    # Write history to json file
    with open(os.path.join(logpath, 'log.' + label + '.json'), 'w') as fp:
        json.dump(history, fp, indent=True)


def SaveConfigParserArguments(tmp_config, savePath):
    with open(f'{savePath}/{tmp_config["training.SelfSupervisedModel"]}.json', 'w') as f:
        json.dump(tmp_config, f, indent=2)


def trainunet(config, number_of_classes, class_weights, label=None):
    """
    trainunet: Create and train a network based on the parameters contained within the configuration file
        - create a directory named 'graphs' in which the training and validation history are stored
        - create a directory named 'models' in which the trained network is saved

    :param config: contains all the configuration file arguments.
    :param number_of_classes: (int)  the number of classes used in the data
    :param class_weights: list of (int) the weight of each classes put in the list format (currently unused)
    :return: (string) the label of the network
    """

    if not label:
        label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Delete any results if network label already exists
    filePath = filepath_utils.FilepathGenerator(config)
    delete_model(filePath, config['detector.outputpath'], label)

    # Create output directories
    os.makedirs(os.path.join(config['detector.outputpath'], "models"), exist_ok=True)
    os.makedirs(os.path.join(config['detector.outputpath'], "graphs"), exist_ok=True)

    # Save configuration
    with open(os.path.join(config['detector.outputpath'], 'SSL.' + label + '.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Read configuration
    with open(os.path.join(config['detector.outputpath'], 'SSL.' + label + '.json'), 'r') as f:
        config = json.load(f)

    filePath = filepath_utils.FilepathGenerator(config)

    # check train class labels
    if not check_gt_validity(os.path.join(config['detector.traininputpath'], 'train', 'class_labels.json'),
                             config['extraction.class_definitions']):
        raise ValueError("Training patch class definitions differ from those defined in the configuration file")
    # check valid class labels
    if config['detector.validationinputpath'] is not None: 
        if not check_gt_validity(os.path.join(config['detector.validationinputpath'], 'validation', 'class_labels.json'),
                             config['extraction.class_definitions']):
            raise ValueError("Validation patch class definitions differ from those defined in the configuration file")

    if config['normalisation.normalise_image']:
        shutil.copyfile(os.path.join(config['extraction.extractbasepath'], 'histogram_matching_stats.hdf5'),
                        os.path.join(config['detector.outputpath'], 'models', 'histogram_matching_stats.hdf5'))

    # Configure UNet
    if config['detector.LR_weightdecay'] is not None:
       # import tensorflow_addons as tfa
        # optimiser = tfa.optimizers.AdamW(learning_rate=config['detector.learn_rate'],
        #                                  weight_decay=config['detector.LR_weightdecay'], name='AdamW')
        optimiser = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=config['detector.LR_weightdecay'],
                                              beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='AdamW')
    else:
        optimiser = tf.keras.optimizers.Adam(learning_rate=config['detector.learn_rate'], decay=0.0, name='Adam')

    if config['detector.colour_mode'] == 'rgb':
        input_shape = (config['detector.patch_size'], config['detector.patch_size'], 3)
    else:
        input_shape = (config['detector.patch_size'], config['detector.patch_size'], 1)

    if config['trainingstrategy.strategy'] == 'learncolour':
        learncolour = True
    else:
        learncolour = False

    if config['detector.transferlearning'] :
        pretrained_models = {}
        if 'simclr' in config['transferlearning.pretrained_ssl_model']:
            simclr_model = SimCLR(simclr_model_path=config['transferlearning.simclr_model_path'],
                                  simclr_model_trainable=config['transferlearning.pretrained_model_trainable'],
                                  inp_shape=input_shape, depth=config['detector.network_depth'],
                                  filter_factor_offset=config['detector.filter_factor_offset'],
                                  initialiser=config['detector.weightinit'], padding=config['detector.padding'],
                                  batchnormalisation=config['detector.batchnormalisation'],
                                  k_size=config['detector.kernel_size'], dropout=config['detector.dropout'])
            pretrained_models['simclr_model'] = simclr_model

        elif 'byol' in config['transferlearning.pretrained_ssl_model']:
            byol_model = BYOL(byol_model_path=config['transferlearning.byol_model_path'],
                              byol_model_trainable=config['transferlearning.pretrained_model_trainable'],
                              inp_shape=input_shape, depth=config['detector.network_depth'],
                              filter_factor_offset=config['detector.filter_factor_offset'],
                              initialiser=config['detector.weightinit'], padding=config['detector.padding'],
                              batchnormalisation=config['detector.batchnormalisation'],
                              k_size=config['detector.kernel_size'], dropout=config['detector.dropout'])
            pretrained_models['byol_model'] = byol_model

        else:
            raise ValueError("Pretraining model is not correct. It should be one of ['simclr', 'byol']")

        UNet, inp_shape, otp_shape = unet_decoder_above_SSL(pretrained_models,
                                                            inp_shape=input_shape, nb_classes=number_of_classes,
                                                            depth=config['detector.network_depth'],
                                                            filter_factor_offset=config['detector.filter_factor_offset'],
                                                            initialiser=config['detector.weightinit'],
                                                            padding=config['detector.padding'],
                                                            modifiedarch=config['detector.modifiedarch'],
                                                            batchnormalisation=config['detector.batchnormalisation'],
                                                            k_size=config['detector.kernel_size'],
                                                            dropout=config['detector.dropout'],
                                                            learnupscale=config['detector.learnupscale'])
    else:
        pretrained_models = None
        UNet, inp_shape, otp_shape = baseline_unet_model(input_shape, number_of_classes,
                                                         depth=config['detector.network_depth'],
                                                         filter_factor_offset=config['detector.filter_factor_offset'],
                                                         initialiser=config['detector.weightinit'],
                                                         padding=config['detector.padding'],
                                                         modifiedarch=config['detector.modifiedarch'],
                                                         batchnormalisation=config['detector.batchnormalisation'],
                                                         k_size=config['detector.kernel_size'],
                                                         dropout=config['detector.dropout'],
                                                         learnupscale=config['detector.learnupscale'],
                                                         learncolour=learncolour)

    if config['detector.weight_samples']:
        cattarget = True
    else:
        cattarget = False

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
        augmentationparameters['colour_transfer_staindatadir'] = config['general.datapath']

        print(f"ImageDataGenerator with {config['augmentation.methods']} augmentations...")
        train_gen = ImageDataGenerator(methods=config['augmentation.methods'],
                                       augmentationparameters=augmentationparameters,
                                       fill_mode='reflect',
                                       standardise_sample=config['normalisation.standardise_patches'],
                                       samplewise_normalise=config['normalisation.normalise_patches'],
                                       nb_classes=number_of_classes,
                                       categoricaltarget=cattarget)


    else:
        print(f"ImageDataGenerator without any augmentations...")
        train_gen = ImageDataGenerator(standardise_sample=config['normalisation.standardise_patches'],
                                       samplewise_normalise=config['normalisation.normalise_patches'],
                                       nb_classes=number_of_classes,
                                       categoricaltarget=cattarget)

    if config['normalisation.normalise_patches']:
        train_flow = train_gen.fit_and_flow_from_directory(os.path.join(config['detector.traininputpath'], 'train'),
                                                           filepath=filePath,
                                                           img_target_size=(inp_shape[0], inp_shape[1]),
                                                           gt_target_size=(otp_shape[0], otp_shape[1]),
                                                           color_mode=config['detector.colour_mode'],
                                                           batch_size=config['detector.batch_size'],
                                                           shuffle=True,
                                                           augmentationclassblock=augmentationclassblock,
                                                           pretrained_models=pretrained_models)
        # live_augmentation.py is adjusted with Pretrained model Parameters to load data statistics.
        mean, stddev = train_gen.get_fit_stats()
    else:
        train_flow = train_gen.flow_from_directory(os.path.join(config['detector.traininputpath'], 'train'),
                                                   filepath=filePath,
                                                   img_target_size=(inp_shape[0], inp_shape[1]),
                                                   gt_target_size=(otp_shape[0], otp_shape[1]),
                                                   color_mode=config['detector.colour_mode'],
                                                   batch_size=config['detector.batch_size'],
                                                   shuffle=True,
                                                   augmentationclassblock=augmentationclassblock,
                                                   pretrained_models=pretrained_models)
        mean = None
        stddev = None

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
                                        categoricaltarget=cattarget)

    if config['detector.validationinputpath'] is not None: 
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
                                                        pretrained_models=pretrained_models)

    if config['normalisation.normalise_patches']:
        stats = numpy.array([mean, stddev])
        print("Mean: ", stats[0])
        print("Std: ", stats[1])
        # Write normalisation statistics
        os.makedirs(os.path.join(config['detector.outputpath'], "models"), exist_ok=True)
        statsfilename = os.path.join(config['detector.outputpath'], "models", "normalisation_stats." + label + ".hdf5")
        print("Writing in " + statsfilename)
        with h5py.File(statsfilename, "w") as f:
            f.create_dataset("stats", data=stats)

    callbacklist = []
    model_filename = os.path.join(config['detector.outputpath'], "models",
                                  f"{config['detector.segmentationmodel']}_best." + label + ".keras")
    # model_filename_latest = os.path.join(config['detector.outputpath'], "models",
    #                                      config['detector.segmentationmodel'] + "_{epoch:02d}." + label + ".hdf5")
    model_filename_latest = os.path.join(config['detector.outputpath'], "models",
                                         f"{config['detector.segmentationmodel']}_latest." + label + ".keras")

    if config['detector.validationinputpath'] is not None: 
        callbacklist.append(ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'))
    else:
        callbacklist.append(ModelCheckpoint(model_filename, monitor='loss', verbose=1, save_best_only=True, mode='auto'))
        
    callbacklist.append(ModelCheckpoint(model_filename_latest, verbose=1, save_freq=int(1 * len(train_flow))))

    historyfile = os.path.join(config['detector.outputpath'], "models",
                               f"{config['detector.segmentationmodel']}_history." + label + ".json")
    
    callbacklist.append(SaveHistory(historyfile, read_existing=False))

    if config['detector.earlyStopping']:
        callbacklist.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'))

    callbacklist.append(DisplayLearningRateValue())

    if config['detector.reducelr']:
        # callbacklist.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
        #                                       min_delta=0.0001, cooldown=0, min_lr=0))
        callbacklist.append(ReduceLearningRate(patience=config['detector.reduceLR_patience'], factor=0.1, verbose=1))

    if config['detector.weight_samples']:
        weights = train_gen.get_weights()
        loss_fun = weighted_categorical_crossentropy(weights)
        metric = 'categorical_accuracy'
    else:
        loss_fun = 'sparse_categorical_crossentropy'
        metric = 'sparse_categorical_accuracy'
    UNet.compile(optimizer=optimiser, loss=loss_fun, metrics=[metric])
    # UNet.summary()
    
    print("Number of classes: %s" % number_of_classes)
    print("Patch size: %s" % str(input_shape))

    if config['detector.reducelr']:
        print(f"LR: {config['detector.learn_rate']} will be reduced with a factor of 0.1 at "
              f"{config['detector.reduceLR_percentile']}-th percentile ({config['detector.reduceLR_patience']}-th "
              f"epoch) of training.")
    else:
        print(f"LR: {config['detector.learn_rate']}")

    with open(os.path.join(config['detector.outputpath'], "config_value." + label + ".txt"), 'w') as f:
        f.write(
            "loss: " + str(UNet.loss) + '\n' + "metrics: " + str(UNet.metrics) + '\n' + "function: " + str(UNet) + '\n')

    with open(os.path.join(config['detector.outputpath'], "models", "summary." + label + ".txt"), 'w') as f:
        UNet.summary(print_fn=lambda x: f.write(x + '\n'))

    if config['detector.validationinputpath'] is not None:
        UNet.fit(train_flow, epochs=config['detector.epochs'], shuffle=True, callbacks=callbacklist,
                 validation_data=valid_flow, verbose=2)#, use_multiprocessing=True, workers=12, max_queue_size=650)
    else:
        UNet.fit(train_flow, epochs=config['detector.epochs'], shuffle=True, callbacks=callbacklist,
                 verbose=2)#, use_multiprocessing=True, workers=12, max_queue_size=650)
        
    __save_training_history(os.path.join(config['detector.outputpath'], "graphs"), historyfile, label, config['detector.validation_data'])

    print("Written model with label: %s" % label)

    # if os.path.exists(model_filename_latest):
    #     os.remove(model_filename_latest)

    return label


def pretrained_simclr_model_path(conf):
    return os.path.join(conf['general.workpath'], conf['general.additionalpath'], 'saved_models/postdoc',
                        'improve_kidney_glomeruli_segmentation/sysmifta/pretraining/final_selected_models',
                        'simclr/simclr_unet_encoder.h5')

def pretrained_byol_model_path(conf):
    return os.path.join(conf['general.workpath'], conf['general.additionalpath'], 'saved_models/postdoc',
                        'improve_kidney_glomeruli_segmentation/sysmifta/pretraining/final_selected_models',
                        'byol/byol_unet_encoder.h5')

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

    elif conf['detector.validation_data'].lower() == "respective_splits":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], f'epochs_{conf["detector.epochs"]}',
                                                  'downstream_tasks_evaluation',
                                                  conf['detector.patchstrategy'], conf['detector.percentN'],
                                                  conf['general.staincode'], 'respective_splits_validation_data')

    elif conf['detector.validation_data'].lower() == "full":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], f'epochs_{conf["detector.epochs"]}',
                                                  'downstream_tasks_evaluation',
                                                  conf['detector.patchstrategy'], conf['detector.percentN'],
                                                  conf['general.staincode'], 'full_validation_data')

    if conf['detector.transferlearning']:
        conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model.lower()
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable

        if 'simclr' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.simclr_model_path'] = pretrained_simclr_model_path(conf)
        elif 'byol' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.byol_model_path'] = pretrained_byol_model_path(conf)
        else:
            raise ValueError(
                "Self-supervised learning based pretrained-models should be one of ['simclr', 'byol', 'hrcsco']")

        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'],
                                                   conf['transferlearning.pretrained_ssl_model'])
    else:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'])

    conf['segmentation.segmentationpath'] = os.path.join(conf['detector.outputpath'], conf['segmentation.segmentationpath'])
    conf['segmentation.detectionpath'] = os.path.join(conf['detector.outputpath'], conf['segmentation.detectionpath'])
    return conf



def setreproducibility():
    import tensorflow as tf
    import random as rn

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    numpy.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train.')

    parser.add_argument('-c', '--configfile', type=str, default='configuration_files/finetune/hubmap.cfg')
    parser.add_argument('-l', '--label', type=str, help='saved model name', default='test_finetune_unet_hubmap')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
    parser.add_argument('-r', '--reproducible', action='store_const', default=False, const=True,
                        help='set seeds to give reproducible results')

    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, default='simclr', help='None, simclr | byol | hrcsco')
    parser.add_argument('-pmt', '--pretrained_model_trainable', default=True,
                        type=lambda x: str(x).lower() in ['true', '1', 'yes'], help='if finetune: True | '
                                                                                    'if fixedfeatures: False')

    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pn', '--percent_N', type=str, default='percent_1', help='percent_1 | percent_5 | percent_10 '
                                                                                    '| percent_20 | percent_50 | '
                                                                                    'percent_100')

    parser.add_argument('-lr', '--LR', type=str, default='0.0001')
    parser.add_argument('-lrd', '--LR_weightdecay', type=str, default="None")
    parser.add_argument('-rlrp', '--reduceLR_percentile', type=str, default='90', help='percentile to reduce LR')
    
    parser.add_argument('-vd', '--validation_data', type=str, default='full', help="none | respective_splits | full")
    parser.add_argument('-nte', '--num_training_epochs', type=int, default=250)

     

    args = parser.parse_args()

    if args.reproducible:
        setreproducibility()

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
        if config['detector.reducelr']:
            config['detector.reduceLR_percentile'] = int(args.reduceLR_percentile)
            config['detector.reduceLR_patience'] = int((config['detector.reduceLR_percentile'] / 100) * config['detector.epochs'])


    number_of_classes = len(config['extraction.class_definitions'])
    class_weights = [1] * number_of_classes

    print('Command Line Arguments:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args
    print("\nConfiguration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))
    
    start = datetime.datetime.now()
    if not os.path.isfile(os.path.join(config['detector.outputpath'], 'graphs', 'loss_history.' + args.label + '.png')):
        trainunet(config, number_of_classes, class_weights, label=args.label)
    else:
        print("Model is already trained...!")
    print('\nTraining Time: {}'.format(datetime.datetime.now() - start))
