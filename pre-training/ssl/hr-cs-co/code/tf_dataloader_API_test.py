import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import json
import copy
import random
import numpy
import staintools
import multiprocessing.pool
import albumentations as A
import tensorflow as tf
import tensorflow.keras.backend as K

from datetime import datetime
from glob import glob
from dataloader.pillow_API import load_img
from dataloader.augmentations import which_custom_augmentations
import config_utils
from plot import stain_separation_plot, images_with_labels_plot, images_with_labels_plus_augmentations_plot


def standardise(image):
    return (image - image.min()) / ((image.max() - image.min()) + K.epsilon())


def Compute_Mean_Std_Dataset(config, filenames):
    print(f"Computing mean and stddev over stain {config['data.Stain']} dataset ({len(filenames)} images)...")
    start_time = datetime.now()
    random_transforms = A.Compose(which_custom_augmentations(config=config))

    def get_mean_std(filepath):
        x = load_img(filepath, grayscale=False, target_size=(config["training.CroppedImageSize"][0],
                                                             config["training.CroppedImageSize"][1]), data_format=None)

        if config["preprocessing.AugmentInComputeMeanStd"]:
            if random.random() < 0.5:
                x = random_transforms(image=x)['image']

        x = standardise(x)
        tmp_mean = numpy.mean(x)
        tmp_var = numpy.var(x)

        return tmp_mean, tmp_var, x.size

    def mean_std_unpack(args):
        _, filepath = args[0], args[1]
        return get_mean_std(filepath)

    pool = multiprocessing.pool.ThreadPool()
    samples = pool.map(mean_std_unpack, ((i, filepath) for i, filepath in enumerate(filenames)))

    mean = 0
    var = 0
    k = 0

    for batch_mean, batch_var, s in samples:
        old_mean = mean

        mean = (k * 1.0 / (k + s)) * mean + (s * 1.0 / (k + s)) * batch_mean
        var = (k * 1.0 / (k + s)) * var + \
              (s * 1.0 / (k + s)) * batch_var + \
              ((k * s) * 1.0 / ((k + s) * (k + s))) * ((old_mean - batch_mean) * (old_mean - batch_mean))

        k += s

    std = numpy.sqrt(var)
    print(f"\nTime Spent: {datetime.now() - start_time}")
    return mean, std


class TFDataLoader:
    def __init__(self, config, ssl_model_phase, mode="train", print_data_specifications=False, ):
        """
        - Using Command Line arguments
        """
        self.config = config
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.ssl_model_phase = ssl_model_phase
        self.mode = mode
        self.print_data_specifications = print_data_specifications
        if self.ssl_model_phase == "contrastive_learning":
            self.transforms = A.Compose(which_custom_augmentations(config=self.config))

    def GetPNGImageFiles(self, num_examples_mode=False):
        if self.mode == "train":
            filenames = glob(os.path.join(self.config["data.TrainDataDir"], "*.png"))
        elif self.mode == "validation":
            filenames = glob(os.path.join(self.config["data.ValidationDataDir"], "*.png"))

        random.shuffle(filenames)

        if self.print_data_specifications:
            print(f"Found {len(filenames)} {self.mode} images belonging to stain {self.config['data.Stain']}.")
        if num_examples_mode:
            return len(filenames)
        else:
            return filenames

    def PILImageRead_with_StainSeparation(self, path):
        x = load_img(path, grayscale=False, target_size=(self.config["training.CroppedImageSize"][0],
                                                         self.config["training.CroppedImageSize"][1]), data_format=None)
        stain_separator = staintools.StainAugmentor(method=self.config["stain_separation.Method"],
                                                    sigma1=self.config["stain_separation.Perturb_alpha_range"],
                                                    sigma2=self.config["stain_separation.Perturb_beta_range"],
                                                    include_background=False)
        source_concentrations, matrix, zeroed_channels = stain_separator.separate(image=x.astype(numpy.uint8).copy(),
                                                                                  stain_code=self.config["data.Stain"])
        if self.ssl_model_phase == "contrastive_learning":
            perturbed_concentrations, x_perturbed = stain_separator.perturb(image=x.astype(numpy.uint8).copy(),
                                                                            source_concentrations=source_concentrations,
                                                                            stain_matrix=matrix,
                                                                            zeroed_channels=zeroed_channels)

        x = standardise(x.astype(numpy.float32).copy())
        source_concentrations = standardise(source_concentrations.reshape(x.shape).astype(numpy.float32).copy())

        if self.ssl_model_phase == "contrastive_learning":
            perturbed_concentrations = standardise(perturbed_concentrations.reshape(x.shape).astype(numpy.float32).copy())
            x_perturbed = standardise(x_perturbed.reshape(x.shape).astype(numpy.float32))
        if self.config["data.Stain"] != "03":
            H_channel, O_channel = source_concentrations[:, :, 0], source_concentrations[:, :, 1]
        else:
            H_channel, O_channel = source_concentrations[:, :, 0], source_concentrations[:, :, 1] + source_concentrations[:, :, 2]

        if self.ssl_model_phase == "contrastive_learning":
            if self.config["data.Stain"] != "03":
                H_prime_channel, O_prime_channel = perturbed_concentrations[:, :, 0], perturbed_concentrations[:, :, 1]
            else:
                H_prime_channel, O_prime_channel = perturbed_concentrations[:, :, 0], perturbed_concentrations[:, :, 1] + perturbed_concentrations[:, :, 2]

        H_label = copy.deepcopy(H_channel)
        O_label = copy.deepcopy(O_channel)
        if self.ssl_model_phase == "contrastive_learning":
            H_prime_label = copy.deepcopy(H_prime_channel)
            O_prime_label = copy.deepcopy(O_prime_channel)

        if self.ssl_model_phase == "contrastive_learning":
            return x, x_perturbed, tf.expand_dims(H_channel, axis=-1), tf.expand_dims(O_label, axis=-1),\
                   tf.expand_dims(O_channel, axis=-1), tf.expand_dims(H_label, axis=-1),\
                   tf.expand_dims(H_prime_channel, axis=-1), tf.expand_dims(O_prime_label, axis=-1),\
                   tf.expand_dims(O_prime_channel, axis=-1), tf.expand_dims(H_prime_label, axis=-1)
        else:
            return x, tf.expand_dims(H_channel, axis=-1), tf.expand_dims(O_label, axis=-1), \
                   tf.expand_dims(O_channel, axis=-1), tf.expand_dims(H_label, axis=-1)

    def ParseImages(self, paths):
        if self.ssl_model_phase == "contrastive_learning":
            return tf.numpy_function(func=self.PILImageRead_with_StainSeparation, inp=[paths],
                                     Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                           tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        else:
            return tf.numpy_function(func=self.PILImageRead_with_StainSeparation, inp=[paths],
                                     Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

    def AugmentFunction(self, image, image_perturbed, image_H, label_O, image_O, label_H,
                        image_H_prime, label_O_prime, image_O_prime, label_H_prime):
        H2O_T1 = self.transforms(image=image_H, mask=label_O)
        O2H_T1 = self.transforms(image=image_O, mask=label_H)

        H2O_T2 = self.transforms(image=image_H_prime, mask=label_O_prime)
        O2H_T2 = self.transforms(image=image_O_prime, mask=label_H_prime)

        return image, image_perturbed, H2O_T1["image"], H2O_T1["mask"], O2H_T1["image"], O2H_T1["mask"], \
               H2O_T2["image"], H2O_T2["mask"], O2H_T2["image"], O2H_T2["mask"]

    def ProcessAugmentations(self, image, image_perturbed, image_H, label_O, image_O, label_H,
                             image_H_prime, label_O_prime, image_O_prime, label_H_prime):
        return tf.numpy_function(func=self.AugmentFunction,
                                 inp=[image, image_perturbed, image_H, label_O, image_O, label_H,
                                      image_H_prime, label_O_prime, image_O_prime, label_H_prime],
                                 Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                       tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

    def AugmentationReshape(self, image, image_perturbed, image_H_T1, label_O_T1, image_O_T1, label_H_T1,
                            image_H_T2, label_O_T2, image_O_T2, label_H_T2):
        image_H_T1.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))
        label_O_T1.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))
        image_O_T1.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))
        label_H_T1.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))
        image_H_T2.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))
        label_O_T2.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))
        image_O_T2.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))
        label_H_T2.set_shape((self.config["training.CroppedImageSize"][0],
                              self.config["training.CroppedImageSize"][1], 1))

        return image, image_perturbed, image_H_T1, label_O_T1, image_O_T1, label_H_T1, \
               image_H_T2, label_O_T2, image_O_T2, label_H_T2

    def PerformanceConfiguration(self, ds):
        buffer_multiplier = 50 if self.config["training.CroppedImageSize"][0] <= 32 else 10
        ds = ds.shuffle(self.config["training.BatchSize"] * buffer_multiplier)
        ds = ds.batch(self.config["training.BatchSize"], drop_remainder=False)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def LoadDataset(self):
        filenames_ds = tf.data.Dataset.from_tensor_slices(self.GetPNGImageFiles(num_examples_mode=False))
        ds = filenames_ds.map(self.ParseImages, num_parallel_calls=self.AUTOTUNE)
        if self.ssl_model_phase == "contrastive_learning":
            ds = ds.map(self.ProcessAugmentations, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
            ds = ds.map(self.AugmentationReshape, num_parallel_calls=self.AUTOTUNE)
        ds = self.PerformanceConfiguration(ds)
        return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ChecK validity of TF_DataLoader API...!")
    parser.add_argument('-c', '--configFile', type=str, default="configuration_files/CS_CO_test.cfg", help='cfg file to use')
    parser.add_argument('-m', '--deepModel', type=str, default="UNet", help='network architecture to use')
    parser.add_argument("-r", "--repetition", type=str, default="rep1")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("\n")
    print("Gpu Growth Restriction Done...")
    print("\n")

    args = parser.parse_args()

    if args.configFile:
        config = config_utils.readconfig(args.configFile)
    else:
        config = config_utils.readconfig()

    print('Command Line Input Arguments:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    print("\n")

    if args.deepModel.lower() == "unet" or args.deepModel.lower() == "resnet50":
        config["training.DeepModel"] = args.deepModel.lower()
    else:
        raise ValueError("Please specify one of (\'UNet, ResNet50\') in parser.parse_args().DeepModel.")

    if "rep" in args.repetition:
        config["training.repetition"] = args.repetition
    else:
        raise ValueError("Please specify proper repetition in parser.parse_args().repetition.")

    config["output.DataStatsDir"] = os.path.join(config["output.OutputDir"], config["training.DeepModel"],
                                                 config["output.Label"])
    config["output.OutputDir"] = os.path.join(config["output.OutputDir"], config["training.DeepModel"],
                                              config["output.Label"], config["training.repetition"],
                                              config["training.SSLModelPhase"])
    os.makedirs(config["output.OutputDir"], exist_ok=True)

    print("\nConfiguration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))
    ds_train = TFDataLoader(config, ssl_model_phase=config["training.SSLModelPhase"],
                            mode="train", print_data_specifications=True).LoadDataset()

    if config["training.SSLModelPhase"] == "cross_stain_prediction":
        for step, (x, images_H, labels_O, images_O, labels_H) in enumerate(ds_train.take(1)):
            stain_separation_plot(x, images_H, images_O, config['output.OutputDir'], config['data.Stain'])
            images_with_labels_plot(images_H, labels_O, images_O, labels_H, output_dir=config['output.OutputDir'],
                                    stain_code=config['data.Stain'])

    elif config["training.SSLModelPhase"] == "contrastive_learning":
        for step, (x, x_perturbed, images_H_T1, labels_O_T1, images_O_T1, labels_H_T1,
                   images_H_T2, labels_O_T2, images_O_T2, labels_H_T2) in enumerate(ds_train.take(1)):
            images_with_labels_plus_augmentations_plot(x, x_perturbed, images_H_T1, labels_O_T1, images_O_T1, labels_H_T1,
                                                       images_H_T2, labels_O_T2, images_O_T2, labels_H_T2,
                                                       output_dir=config['output.OutputDir'],
                                                       stain_code=config['data.Stain'])

    print(f"Sample Data outputs are saved in {config['output.OutputDir']}")
    del ds_train

