from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from utils import image_utils,data_utils
import os
import numpy
import shutil
import json


# State monitor callback. Tracks how well we are doing and writes
# some state to a json file. This lets us resume training seamlessly.
#
# ModelState.state is:
#
# { "epoch_count": nnnn,
#   "best_values": { dictionary with keys for each log value },
#   "best_epoch": { dictionary with keys for each log value }
# }

class ModelState(Callback):

    def __init__(self, state_path, read_existing=False, epoch=0):

        self.state_path = state_path

        if read_existing and os.path.isfile(state_path):
            print('Loading existing .json state')
            with open(state_path, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {'epoch_count': epoch,
                          'best_values': {},
                          'best_epoch': {}
                          }

    def on_epoch_end(self, epoch, logs={}):

        # Currently, for everything we track, lower is better

        for k in logs:
            if k not in self.state['best_values'] or logs[k] < self.state['best_values'][k]:
                self.state['best_values'][k] = float(logs[k])
                self.state['best_epoch'][k] = epoch

        self.state['epoch_count'] = epoch+1

        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=4)


class SaveHistory(Callback):

    def __init__(self, history_path, read_existing=False):

        self.history_path = history_path

        if read_existing and os.path.isfile(history_path):
            print('Loading existing .json history')
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        for k in logs:
            if k in self.history:
                self.history[k].append(logs[k])
            else:
                self.history[k] = [logs[k]]

        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


class CheckpointTrainingPatches(Callback):

    def __init__(self, source_dir, patches_dir, inp_shape, otp_shape, colour_mode, mean=None, stddev=None):
        self.inp_shape = inp_shape
        self.otp_shape = otp_shape
        self.mean = mean
        self.stddev = stddev
        self.source_images_test, self.source_masks_test, self.source_class_number, self.source_filenames = self.__getdataset__(source_dir, colour_mode, True, 8)
        self.output_path = patches_dir

    def __getdataset__(self, input_patch_path, colour_mode, standardise_patches=False, number_of_patches_per_class=8):
        # Read images from directory
        classnames = [x[1] for x in os.walk(os.path.join(input_patch_path, 'images'))][0]
        class_number = len(classnames)
        images_test = []
        masks_test = []
        filenames = []
        for classname in classnames:
            dirlist = os.listdir(os.path.join(input_patch_path, 'images', classname))

            perm = numpy.random.permutation(len(dirlist))
            randomdirlist = [dirlist[i] for i in perm[:number_of_patches_per_class]]

            for filename in randomdirlist:
                image = image_utils.read_image(os.path.join(input_patch_path, 'images', classname, filename)).astype(
                    numpy.float32)
                print(filename)
                image = image_utils.image_colour_convert(image, colour_mode)
                inp_diff = numpy.subtract(list(image.shape[:-1]), list(self.inp_shape[:-1]))
                inp_diff //= 2

                image = image[inp_diff[0]:image.shape[0] - inp_diff[0], inp_diff[1]:image.shape[1] - inp_diff[1], :]
                images_test.append(image)

                mask = image_utils.read_image(os.path.join(input_patch_path, 'gts', classname, filename)).astype(
                    numpy.float32)
                mask = numpy.expand_dims(mask, axis=2)
                otp_diff = numpy.subtract(list(mask.shape[:-1]), list(self.otp_shape[:-1]))
                otp_diff //= 2

                mask = mask[otp_diff[0]:mask.shape[0] - otp_diff[0], otp_diff[1]:mask.shape[1] - otp_diff[1], :]

                masks_test.append(mask)

                filenames.append(filename)

        images_test = numpy.array(images_test)
        masks_test = numpy.array(masks_test)

        if standardise_patches:
            for idx, sample in enumerate(images_test):
                images_test[idx, ] = data_utils.standardise_sample(images_test[idx, ])

        # Normalise data
        if self.mean and self.stddev:
            for idx, sample in enumerate(images_test):
                images_test[idx,] = data_utils.normalise_sample(images_test[idx, ], self.mean, self.stddev)

        return images_test, masks_test, class_number, filenames

    def __predictions__(self, images_test, filenames, class_number, model, output_path, epoch):
        y_pred = model.predict(images_test.astype(model.inputs[0].dtype.name))

        img_output_path = os.path.join(output_path, 'predictions', str(epoch))
        if os.path.exists(img_output_path):
            shutil.rmtree(img_output_path, ignore_errors=True)
        os.makedirs(img_output_path)

        for pred, filename in zip(y_pred, filenames):
            for c in range(class_number):
                image_utils.save_image((pred[..., c]*255).astype(numpy.uint8), os.path.join(img_output_path, os.path.splitext(os.path.basename(filename))[0] + '_' + str(c) + '.png'))

    def on_epoch_end(self, epoch, logs):
        self.__predictions__(self.source_images_test, self.source_filenames, self.source_class_number, self.model, self.output_path, epoch)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

class ReduceLearningRate(Callback):
    def __init__(self, patience, factor, verbose):
        self.patience = patience
        self.factor = factor
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.patience:
            old_lr = K.get_value(self.model.optimizer.learning_rate)
            new_lr = old_lr * self.factor
            if self.verbose == 1:
                print(f"Epoch:{epoch + 1}, LR is adjusted from {format(old_lr, '.8f')} to {format(new_lr, '.8f')}\n")

            self.model.optimizer.learning_rate.assign(new_lr)


class DisplayLearningRateValue(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"current_LR: {format(self.model.optimizer.learning_rate.numpy().astype('float32'), '.8f')} at Epoch: {epoch+1}\n")