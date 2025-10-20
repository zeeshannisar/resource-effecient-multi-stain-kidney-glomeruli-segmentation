# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
from train_udagan import getdataset
import os
from utils import config_utils
from unet import unet_models2
from tensorflow.keras.optimizers import Adam

class KerasBatchClassifier(KerasClassifier):

    def fit(self, config, **kwargs):

        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit ###################################################
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__

        ################################################################################################################

        augmentationparameters = {}
        augmentationparameters['affine_rotation_range'] = self.config['augmentation.affine_rotation_range']
        augmentationparameters['affine_width_shift_range'] = self.config['augmentation.affine_width_shift_range']
        augmentationparameters['affine_height_shift_range'] = self.config['augmentation.affine_height_shift_range']
        augmentationparameters['affine_rescale'] = self.config['augmentation.affine_rescale']
        augmentationparameters['affine_zoom_range'] = self.config['augmentation.affine_zoom_range']
        augmentationparameters['affine_horizontal_flip'] = self.config['augmentation.affine_horizontal_flip']
        augmentationparameters['affine_vertical_flip'] = self.config['augmentation.affine_vertical_flip']
        augmentationparameters['elastic_sigma'] = self.config['augmentation.elastic_sigma']
        augmentationparameters['elastic_alpha'] = self.config['augmentation.elastic_alpha']
        augmentationparameters['smotenneighbours'] = self.config['augmentation.smotenneighbours']

        train_gen = ImageDataGenerator(methods=self.config['augmentation.methods'],
                                       augmentationparameters=augmentationparameters,
                                       fill_mode='reflect',
                                       standardise_sample=self.config['normalisation.standardise_patches'],
                                       samplewise_normalise=True,
                                       nb_classes=number_of_classes,
                                       categoricaltarget=False,
                                       validation_split=self.config['detector.validation_fraction'])

        train_flow = train_gen.fit_and_flow_from_directory(os.path.join(self.config['detector.inputpath'], 'train'),
                                                           img_target_size=(inp_shape[0], inp_shape[1]),
                                                           gt_target_size=(otp_shape[0], otp_shape[1]),
                                                           color_mode=self.config['detector.colour_mode'],
                                                           batch_size=self.config['detector.batch_size'],
                                                           shuffle=True,
                                                           subset='training')

        mean, stddev = train_gen.get_fit_stats()

        validation_gen = ImageDataGenerator(standardise_sample=self.config['normalisation.standardise_patches'],
                                            samplewise_normalise=True,
                                            nb_classes=number_of_classes,
                                            categoricaltarget=False,
                                            validation_split=self.config['detector.validation_fraction'])

        valid_flow = validation_gen.flow_from_directory(os.path.join(self.config['detector.inputpath'], 'train'),
                                                        img_target_size=(inp_shape[0], inp_shape[1]),
                                                        gt_target_size=(otp_shape[0], otp_shape[1]),
                                                        color_mode=self.config['detector.colour_mode'],
                                                        batch_size=self.config['detector.batch_size'], shuffle=True,
                                                        subset='validation',
                                                        dataset_mean=mean,
                                                        dataset_std=stddev)

        self.__history = self.model.fit_generator(
            train_flow,
            epochs=epochs,
            shuffle=True,
            callbacks=callbacklist,
            validation_data=valid_flow)

        return self.__history

    def score(self, X, y, **kwargs):
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)
        outputs = self.model.evaluate(X, y, **kwargs)
        if type(outputs) is not list:
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise Exception('The model is not configured to compute accuracy. '
                        'You should pass `metrics=["accuracy"]` to '
                        'the `model.compile()` method.')

    @property
    def history(self):
        return self.__history


# Function to create model, required for KerasClassifier
def create_model(input_shape, number_of_classes, learn_rate=0.0001, useweights=True, depth=5, filter_factor_offset=0, weightinit='he_normal', padding='same', modifiedarch=False, batchnormalisation='before', kernel_size=3, dropout=False, learnupscale=False):
    # create model
    model = unet_models2.build_UNet(input_shape, number_of_classes, depth=depth, filter_factor_offset=filter_factor_offset, initialiser=weightinit, padding=padding, modifiedarch=modifiedarch, batchnormalisation=batchnormalisation, k_size=kernel_size, dropout=dropout, learnupscale=learnupscale)
    adam_optimiser = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # Compile model
    if useweights:
        model.compile(optimizer=adam_optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'],
                     sample_weight_mode="temporal")
    else:
        model.compile(optimizer=adam_optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
config = config_utils.readconfig("sysmifta_Nx_16_rgb_base.cfg")

patch_input_path = os.path.join(config['detector.inputpath'], 'train')

images_train, masks_train, images_validation, masks_validation, class_number, class_weights  = getdataset(patch_input_path, config['detector.validation_fraction'])
# create model
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=16, verbose=1)
#model = KerasBatchClassifier(build_fn=create_model, epochs=10, batch_size=16, verbose=0)

# define the grid search parameters
batch_size = [1, 2, 4, 8, 16, 32]
epochs = [5, 10, 20, 40, 50, 100]
learn_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
useweights = [False]
depth = [2, 3, 4, 5, 6, 7]
filter_factor_offset = [0, 1, 2, 3]
modifiedarch = [True, False]
weightinit = ['he_normal', 'glorot_uniform']
padding = ['same']
batchnormalisation = ['off', 'before', 'after']
kernel_size = [3, 5, 7]
dropout = [True, False]
learnupscale = [True, False]

input_shape = (config['detector.patch_size'], config['detector.patch_size'], 3)

param_grid = dict(input_shape=input_shape, number_of_classes=[class_number], batch_size=batch_size, learn_rate=learn_rate, epochs=epochs, useweights=useweights, filter_factor_offset=filter_factor_offset, weightinit=weightinit, modifiedarch=modifiedarch, padding=padding, batchnormalisation=batchnormalisation, kernel_size=kernel_size, dropout=dropout, learnupscale=learnupscale)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=1, n_iter=1000, cv=3, verbose=10, error_score=0)

#grid_result = grid.fit(config, verbose=1)
grid_result = grid.fit(images_train, masks_train, verbose=1)

# summarise results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
