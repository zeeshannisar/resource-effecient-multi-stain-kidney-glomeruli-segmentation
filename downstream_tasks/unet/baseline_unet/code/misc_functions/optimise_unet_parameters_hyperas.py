from augmentation.live_augmentation import ImageDataGenerator
from hyperas import optim
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe
from hyperas.distributions import choice
import os
from utils import config_utils
from unet import unet_models
from keras.optimizers import Adam
import numpy


def model(train_gen, validation_gen):

    config = config_utils.readconfig("sysmifta_Nx_16_rgb_base.cfg")

    batch_size = {{choice([1, 2, 4, 8, 16, 32])}}
    epochs = {{choice([5, 10, 20, 40, 50, 100])}}
    learn_rate = {{choice([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3])}}
    useweights = {{choice([False])}}
    depth = {{choice([2, 3, 4, 5, 6, 7])}}
    filter_factor_offset = {{choice([0, 1, 2, 3])}}
    modifiedarch = {{choice([True, False])}}
    weightinit = {{choice(['he_normal', 'glorot_uniform'])}}
    padding = {{choice(['same', 'valid'])}}
    batchnormalisation = {{choice(['off', 'before', 'after'])}}
    kernel_size = {{choice([3, 5, 7])}}
    dropout = {{choice([True, False])}}
    learnupscale = {{choice([True, False])}}

    print('Trial Parameter Values')
    print('\t epochs:', epochs)
    print('\t batch_size:', batch_size)
    print('\t learn_rate:', learn_rate)
    print('\t useweights:', useweights)
    print('\t depth:', depth)
    print('\t filter_factor_offset:', filter_factor_offset)
    print('\t modifiedarch:', modifiedarch)
    print('\t weightinit:', weightinit)
    print('\t padding:', padding)
    print('\t batchnormalisation:', batchnormalisation)
    print('\t kernel_size:', kernel_size)
    print('\t dropout:', dropout)
    print('\t learnupscale:', learnupscale)

    # create model
    number_of_classes = len(config['extraction.class_definitions'])

    if config['detector.colour_mode'] == 'rgb':
        input_shape = (config['detector.patch_size'], config['detector.patch_size'], 3)
    else:
        input_shape = (config['detector.patch_size'], config['detector.patch_size'], 1)

    try:
        model, inp_shape, otp_shape = unet_models.build_UNet(input_shape, number_of_classes, depth=depth, filter_factor_offset=filter_factor_offset, initialiser=weightinit, padding=padding, modifiedarch=modifiedarch, batchnormalisation=batchnormalisation, k_size=kernel_size, dropout=dropout, learnupscale=learnupscale)
        adam_optimiser = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    except Exception as e:
        print(e)
        return {'status': STATUS_FAIL}

    train_generator = train_gen.fit_and_flow_from_directory(os.path.join(config['detector.inputpath'], 'train'),
                                                            img_target_size=(inp_shape[0], inp_shape[1]),
                                                            gt_target_size=(otp_shape[0], otp_shape[1]),
                                                            color_mode=config['detector.colour_mode'],
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            subset='training')
    mean, stddev = train_gen.get_fit_stats()
    validation_generator = validation_gen.flow_from_directory(os.path.join(config['detector.inputpath'], 'train'),
                                                              img_target_size=(inp_shape[0], inp_shape[1]),
                                                              gt_target_size=(otp_shape[0], otp_shape[1]),
                                                              color_mode=config['detector.colour_mode'],
                                                              batch_size=batch_size, shuffle=True,
                                                              subset='validation',
                                                              dataset_mean=mean,
                                                              dataset_std=stddev)

    if useweights:
        weights = train_gen.get_weights()
        loss_fun = weighted_categorical_crossentropy(weights)
    else:
        loss_fun = 'sparse_categorical_crossentropy'

    try:
        # Compile model
        model.compile(optimizer=adam_optimiser, loss=loss_fun, metrics=['sparse_categorical_accuracy'])

        model.fit_generator(train_generator,
                            epochs=epochs,
                            shuffle=True,
                            validation_data=validation_generator)

        loss, acc = model.evaluate_generator(generator=validation_generator)

        print('Test loss:', loss)
        print('Test accuracy:', acc)

        return {'loss': loss, 'status': STATUS_OK, 'model': model}
    except Exception as e:
        print(e)
        return {'status': STATUS_FAIL}


def data_generators(train_gen, validation_gen, inp_shape=None, otp_shape=None, batch_size=8):

    config = config_utils.readconfig("sysmifta_Nx_16_rgb_base.cfg")

    if not inp_shape:
        inp_shape = (config['detector.patch_size'], config['detector.patch_size'])

    if not otp_shape:
        otp_shape = (config['detector.patch_size'], config['detector.patch_size'])

    train_generator = train_gen.fit_and_flow_from_directory(os.path.join(config['detector.inputpath'], 'train'),
                                                       img_target_size=(inp_shape[0], inp_shape[1]),
                                                       gt_target_size=(otp_shape[0], otp_shape[1]),
                                                       color_mode=config['detector.colour_mode'],
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       subset='training')

    mean, stddev = train_gen.get_fit_stats()

    validation_generator = validation_gen.flow_from_directory(os.path.join(config['detector.inputpath'], 'train'),
                                                    img_target_size=(inp_shape[0], inp_shape[1]),
                                                    gt_target_size=(otp_shape[0], otp_shape[1]),
                                                    color_mode=config['detector.colour_mode'],
                                                    batch_size=batch_size, shuffle=True,
                                                    subset='validation',
                                                    dataset_mean=mean,
                                                    dataset_std=stddev)

    return train_generator, validation_generator


def data():

    config = config_utils.readconfig("sysmifta_Nx_16_rgb_base.cfg")

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

    number_of_classes = len(config['extraction.class_definitions'])

    train_gen = ImageDataGenerator(methods=config['augmentation.methods'],
                                   augmentationparameters=augmentationparameters,
                                   fill_mode='reflect',
                                   standardise_sample= config['normalisation.standardise_patches'],
                                   samplewise_normalise=True,
                                   nb_classes=number_of_classes,
                                   categoricaltarget=False,
                                   validation_split=config['detector.validation_fraction'])

    #train_generator = train_gen.fit_and_flow_from_directory(os.path.join(config['detector.inputpath'], 'train'),
    #                                                   img_target_size=(config['detector.patch_size'], config['detector.patch_size']),
    #                                                   gt_target_size=(config['detector.patch_size'], config['detector.patch_size']),
    #                                                   color_mode=config['detector.colour_mode'],
    #                                                   batch_size=batch_size,
    #                                                   shuffle=True,
    #                                                   subset='training')

    validation_gen = ImageDataGenerator(standardise_sample=config['normalisation.standardise_patches'],
                                        samplewise_normalise=True,
                                        nb_classes=number_of_classes,
                                        categoricaltarget=False,
                                        validation_split=config['detector.validation_fraction'])

    #mean, stddev = train_gen.get_fit_stats()

    #validation_generator = validation_gen.flow_from_directory(os.path.join(config['detector.inputpath'], 'train'),
    #                                                img_target_size=(config['detector.patch_size'], config['detector.patch_size']),
    #                                                gt_target_size=(config['detector.patch_size'], config['detector.patch_size']),
    #                                                color_mode=config['detector.colour_mode'],
    #                                                batch_size=batch_size, shuffle=True,
    #                                                subset='validation',
    #                                                dataset_mean=mean,
    #                                                dataset_std=stddev)

    #return train_generator, validation_generator
    return train_gen, validation_gen


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# fix random seed for reproducibility
seed = 7

trials = Trials()

best_run, best_model, search_space = optim.minimize(model=model,
                                                    data=data,
                                                    algo=tpe.suggest,
                                                    max_evals=5,
                                                    trials=trials,
                                                    rseed=seed,
                                                    eval_space=True,
                                                    return_space=True)

if not best_model:
    print("No model found")
else:
    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        tmp = {}
        for k,v in list(vals.items()):
            tmp[k] = v[0]
        print("Trial %s vals: %s" % (t, space_eval(search_space, tmp)))

    print("Best parameter combination found:")
    for i in best_run:
        print("\t", i, ": ", best_run[i])

    print("Evaluation of best performing model:")
    train_gen, validation_gen = data()
    inp_shape = best_model.inputs[0].get_shape().as_list()[1:3]
    otp_shape = best_model.outputs[0].get_shape().as_list()[1:3]
    train_generator, validation_generator = data_generators(train_gen, validation_gen, inp_shape, otp_shape)

    print(best_model.evaluate_generator(generator=validation_generator))
