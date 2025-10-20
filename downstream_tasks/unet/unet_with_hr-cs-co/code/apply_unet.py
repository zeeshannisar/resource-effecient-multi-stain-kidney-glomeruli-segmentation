import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import glob
import h5py
import json
from utils import image_utils, config_utils, data_utils, filepath_utils
import numpy
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import epsilon
from skimage.util import view_as_windows, pad, crop
# from unet.losses import weighted_categorical_crossentropy
import sys
from threading import Thread
import tensorflow as tf
from tifffile import memmap
import tempfile
import shutil
import warnings
from background_tissue_detection import _calculate_mean
import skimage.color
import staintools
import random
from augmentation import stain_transform
import datetime
from utils.select_gpu import pick_gpu_lowest_memory


# todo: openslide read region and read whole image give different results for the same image and region! So process from disk and process from memory will give slightly different resutls...

# class NetworkPredictThread(Thread):
#
#    def __init__(self, network, patch_batch, patch_coords):
#        self.prediction = None
#        self.network = network
#        self.patch_batch = patch_batch
#        self.patch_coords = patch_coords
#        super(NetworkPredictThread, self).__init__()
#
#    def run(self):
#        self.prediction = self.network.predict(self.patch_batch)


class ImagePatchReadThread(Thread):

    def __init__(self, svsImage, x, y, lod, sizex, sizey):
        self.svsImage = svsImage
        self.x = x
        self.y = y
        self.sizex = sizex
        self.sizey = sizey
        self.lod = lod
        self.imgpatch = None
        super(ImagePatchReadThread, self).__init__()

    def run(self):
        self.imgpatch = self.svsImage.read_region((self.x, self.y), self.lod, (self.sizex, self.sizey))


def apply_model_from_memory(imageList, config, modelfilename, modellabel, normalisation_filename):
    """

    apply_model perform segmentation for every class using the network

    :param imageList: list of (string)
    :param config: (dictionary) the configuration file used
    :param modelfilename: (string) the name of the U-Net model used for segmentation
    :param modellabel: (string) the label of the trained network
    :param normalisation_filename: (bool) enable normalisation using a background mask
    """
    modelpath = config["detector.outputpath"]
    stride = config['segmentation.stride']
    batch_size = config['segmentation.batch_size']
    outputpath = config['segmentation.segmentationpath']
    standardise_patches = config['normalisation.standardise_patches']

    filePath = filepath_utils.FilepathGenerator(config)

    if normalisation_filename and config['normalisation.normalise_within_tissue'] and sys.version_info[0] != 2:
        # Only if normalising within tissue
        raise RuntimeError("Only Python 2 is supported")

    # weights = numpy.ones(2)

    UNet = load_model(os.path.join(modelpath, 'models',
                                   modelfilename))  # , custom_objects={'loss': weighted_categorical_crossentropy(weights)})

    number_of_classes = int(UNet.outputs[0].shape[-1])

    if config['normalisation.normalise_patches']:
        # Read normalisation statistics
        statsfilename = os.path.join(modelpath, 'models', "normalisation_stats." + modellabel + ".hdf5")
        with h5py.File(statsfilename, "r") as f:
            mean = f["stats"][0]
            stddev = f["stats"][1]

    img_output_path = os.path.join(outputpath, modellabel)
    if not os.path.exists(img_output_path):
        os.makedirs(img_output_path)

    for file in imageList:

        basename = file.split(os.sep)[-1].split(".")[0]

        # Load image and mask
        image = image_utils.read_svs_image_forced(file, config['detector.lod'])
        image = image.astype(UNet.inputs[0].dtype.name)
        if normalisation_filename:
            if config['normalisation.normalise_within_tissue']:
                tissue_mask = image_utils.read_binary_image(filePath.get_background_mask_for_image(basename)[0])
                normalised_image = image_utils.normalise_rgb_image_from_stat_file(image[tissue_mask > 0, :],
                                                                                  normalisation_filename, config[
                                                                                      'normalisation.normalise_within_tissue'])
                image[tissue_mask > 0] = normalised_image
            else:
                image = image_utils.normalise_rgb_image_from_stat_file(image, normalisation_filename,
                                                                       config['normalisation.normalise_within_tissue'])

        print('image filename: {}'.format(file))
        print('image size: {}, {}'.format(image.shape[1], image.shape[0]))

        inp_patch_size = UNet.inputs[0].get_shape().as_list()[1:3]
        otp_patch_size = UNet.outputs[0].get_shape().as_list()[1:3]
        diff = numpy.subtract(inp_patch_size, otp_patch_size)
        diff //= 2

        padding_x = inp_patch_size[1] - (image.shape[1] % inp_patch_size[1])
        padding_y = inp_patch_size[0] - (image.shape[0] % inp_patch_size[0])
        image = pad(image, ((0, padding_y), (0, padding_x), (0, 0)), "symmetric")

        segmentation = numpy.zeros((image.shape[0], image.shape[1], number_of_classes),
                                   dtype=UNet.outputs[0].dtype.name)
        segmentation_count = numpy.zeros((image.shape[0], image.shape[1]), dtype=numpy.uint32)

        if config["segmentation.stain_transfer"]:
            if config["detector.lod"] < 2:
                processinglod = 2
            else:
                processinglod = config["detector.lod"]
            threshold = _calculate_mean(svsImage, processinglod)

        # Use a view on the array to loop over the patches
        image_patches = view_as_windows(image, (inp_patch_size[0], inp_patch_size[1], image.shape[-1]), step=stride)
        if config['detector.colour_mode'] == 'rgb':
            patch_batch = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], image.shape[-1]),
                                      dtype=UNet.outputs[0].dtype.name)
        else:
            patch_batch = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], 1),
                                      dtype=UNet.outputs[0].dtype.name)
        patch_coords = [None] * batch_size
        c = 0

        # Process every data patch
        for pos_x in range(0, image_patches.shape[1]):
            for pos_y in range(0, image_patches.shape[0]):

                patch = imgpatch[i:i + inp_patch_size[0], j:j + inp_patch_size[1]]

                if config['segmentation.stain_transfer']:
                    dtype = patch.dtype
                    greypatch = skimage.color.rgb2gray(numpy.divide(patch, 255)) * 255
                    if numpy.sum(greypatch > threshold) < greypatch.size * 0.9:
                        target_stain = config["general.staincode"]
                        files = []
                        for subdir in sorted(
                                os.listdir(
                                    os.path.join(config['general.datapath'], target_stain, 'downsampledpatches',
                                                 'train', 'images'))):
                            if subdir != 'background':
                                files += [os.path.join(data_dir, target_stain, 'downsampledpatches', 'train', 'images',
                                                       subdir, name) for name in os.listdir(
                                    os.path.join(config['general.datapath'], target_stain, 'downsampledpatches',
                                                 'train', 'images',
                                                 subdir))]
                        target_image_filename = random.choice(files)
                        target_image = image_utils.read_image(target_image_filename)
                        stain_normalizer = staintools.StainNormalizer(method='macenko')
                        try:
                            stain_normalizer.fit(target_image.astype(numpy.uint8))
                            patch = stain_normalizer.transform(patch.astype(numpy.uint8)).astype(dtype)
                        except:
                            pass

                patch = image_utils.image_colour_convert(patch, config['detector.colour_mode'])

                if standardise_patches:
                    patch = data_utils.standardise_sample(patch)

                if normalise_patches:
                    patch = data_utils.normalise_sample(patch, mean, stddev)

                patch_batch[c, :, :, :] = patch

                patch_coords[c] = (pos_y * stride, pos_x * stride)

                c += 1

                if c == batch_size:
                    sys.stdout.write('x: {:^5}, y: {:^5}\r'.format(pos_x * stride, pos_y * stride))

                    preds = UNet.predict(patch_batch)

                    for i, coord in enumerate(patch_coords):
                        segmentation[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                        coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1], :] += preds[i, :, :, :]
                        segmentation_count[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                        coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1]] += 1

                    c = 0

        if c > 0:
            preds = UNet.predict(patch_batch[0:c, :, :, :])

            for i, coord in enumerate(patch_coords[0:c]):
                segmentation[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1], :] += preds[i, :, :, :]
                segmentation_count[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1]] += 1

        segmentation = crop(segmentation, ((0, padding_y), (0, padding_x), (0, 0)))
        segmentation_count = crop(segmentation_count, ((0, padding_y), (0, padding_x)))

        # Save the output prediction for each class
        for className in config['extraction.class_definitions'].keys():
            label = config['extraction.class_definitions'][className][0]
            output_dir = os.path.join(img_output_path, str(className))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            segmentation[:, :, label] = numpy.divide(segmentation[:, :, label], segmentation_count + epsilon())
            image_utils.save_image((segmentation[:, :, label] * 255).astype(numpy.uint8),
                                   os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + '.png'))


def _process_patch(UNet, diff, imgpatch, inp_patch_size, otp_patch_size, stride, batch_size, patch_batch, patch_coords,
                   patchx, patchy, standardise_patches, normalise_patches, mean, stddev, t, c, segmentation,
                   segmentation_count, first_run, staintransfer=False, threshold=None, data_dir=None,
                   patch_batch_H=None, patch_batch_O=None):
    for i in range(0, imgpatch.shape[0] - (inp_patch_size[0] - 1), stride):
        for j in range(0, imgpatch.shape[1] - (inp_patch_size[1] - 1), stride):

            patch = imgpatch[i:i + inp_patch_size[0], j:j + inp_patch_size[1]]

            if staintransfer:
                dtype = patch.dtype
                greypatch = skimage.color.rgb2gray(numpy.divide(patch, 255)) * 255
                if numpy.sum(greypatch > threshold) < greypatch.size * 0.9:
                    target_stain = config["general.staincode"]
                    files = []
                    for subdir in sorted(
                            os.listdir(os.path.join(data_dir, target_stain, 'downsampledpatches', 'train', 'images'))):
                        if subdir != 'background':
                            files += [
                                os.path.join(data_dir, target_stain, 'downsampledpatches', 'train', 'images', subdir,
                                             name) for name in os.listdir(
                                    os.path.join(data_dir, target_stain, 'downsampledpatches', 'train', 'images',
                                                 subdir))]
                    target_image_filename = random.choice(files)
                    target_image = image_utils.read_image(target_image_filename)
                    stain_normalizer = staintools.StainNormalizer(method='macenko')
                    try:
                        stain_normalizer.fit(target_image.astype(numpy.uint8))
                        patch = stain_normalizer.transform(patch.astype(numpy.uint8)).astype(dtype)
                    except:
                        pass

            patch = image_utils.image_colour_convert(patch, config['detector.colour_mode'])
            if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                patch_H, patch_O = stain_transform.separate(patch, staincode=config["general.staincode"])

            if standardise_patches:
                if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                    patch_H, patch_O = data_utils.standardise_sample(patch_H), data_utils.standardise_sample(patch_O)
                else:
                    patch = data_utils.standardise_sample(patch)

            if normalise_patches:
                if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                    patch_H, patch_O = data_utils.normalise_sample(patch_H, mean, stddev), data_utils.normalise_sample(patch_O, mean, stddev)
                else:
                    patch = data_utils.normalise_sample(patch, mean, stddev)

            if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                patch_batch_H[c, :, :, :] = patch_H
                patch_batch_O[c, :, :, :] = patch_O
                patch_batch[c, :, :, :] = patch
            else:
                patch_batch[c, :, :, :] = patch

            patch_coords[c] = (patchy + i, patchx + j)

            c += 1

            if c == batch_size:
                # Non threaded version
                sys.stdout.write('x: {:^6}, y: {:^6}\r'.format(patchx + j, patchy + i))
                if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                    preds = UNet.predict([patch_batch_H, patch_batch_O])
                else:
                    preds = UNet.predict(patch_batch)

                for k, coord in enumerate(patch_coords):
                    segmentation[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                    coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1], :] += preds[k, :, :, :]
                    segmentation_count[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                    coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1]] += 1

                t = None
                first_run = False
                c = 0

    if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
        return segmentation, segmentation_count, patch_batch, patch_batch_H, patch_batch_O, patch_coords, t, c, first_run
    else:
        return segmentation, segmentation_count, patch_batch, patch_coords, t, c, first_run


def apply_model_from_disk(tmpDir, imageList, config, modelfilename, modeltimestamp, normalisation_filename):
    """
    apply_model perform segmentation for every class using the network
    :param imageList: list of (string)
    :param config: (dictionary) the configuration file used
    :param modelfilename: (string) the name of the U-Net model used for segmentation
    :param modeltimestamp: (string) the timestamp of the trained network
    :param normalisation_filename: (bool) enable normalisation using a background mask
    """

    # todo: need to implement mirroring at the end of the image, currently doesnt classify last part of image

    # # This function uses memmaps, on typical images these can eb 40GB in size at lod 1 so we need to check that we wont fill the disk
    # st = os.statvfs(tempfile.gettempdir())
    # mb = st.f_bavail * st.f_frsize / 1024 / 1024 / 1024  # in GBs
    # if mb < 70:
    #     warnings.warn(
    #         "This function writes large amounts of temporary data to disk, there is less then 70GB free on " + tempfile.gettempdir() + ", do you want to continue?")
    #     cont = ''
    #     while cont not in ['yes', 'no']:
    #         cont = input("yes/no > ")
    #     if not cont == "yes":
    #         return

    modelpath = config["detector.outputpath"]
    stride = config['segmentation.stride']
    batch_size = config['segmentation.batch_size']
    outputpath = config['segmentation.segmentationpath']
    standardise_patches = config['normalisation.standardise_patches']

    if normalisation_filename and config['normalisation.normalise_within_tissue'] and sys.version_info[0] != 2:
        # Only if normalising within tissue
        raise RuntimeError("Only Python 2 is supported")

    # weights = numpy.ones(2)

    UNet = load_model(os.path.join(modelpath, 'models', modelfilename))  # , custom_objects={'loss': weighted_categorical_crossentropy(weights)})
    UNet.make_predict_function()  # Required for executing predict in a separate thread

    number_of_classes = UNet.outputs[0].shape[-1]

    if config['normalisation.normalise_patches']:
        # Read normalisation statistics
        statsfilename = os.path.join(modelpath, 'models', "normalisation_stats." + modeltimestamp + ".hdf5")
        with h5py.File(statsfilename, "r") as f:
            mean = f["stats"][0]
            stddev = f["stats"][1]
        print(f"Mean: {mean} --- Stddev: {stddev}")
    else:
        mean = None
        stddev = None

    img_output_path = os.path.join(outputpath, modeltimestamp)
    if not os.path.exists(img_output_path):
        os.makedirs(img_output_path)

    inp_patch_size = UNet.inputs[0].get_shape().as_list()[1:3]
    otp_patch_size = UNet.outputs[0].get_shape().as_list()[1:3]
    diff = numpy.subtract(inp_patch_size, otp_patch_size)
    diff //= 2

    for file in imageList:

        with tempfile.NamedTemporaryFile(dir=tmpDir, suffix='.h5') as tmp_h5, \
                tempfile.NamedTemporaryFile(dir=tmpDir, suffix='.svs') as tmp_svs, \
                tempfile.NamedTemporaryFile(dir=tmpDir, suffix='.tif') as tmp_tif:

            # Used for systems with SSD main drive (faster random access)
            origfilename = file
            file = tmp_svs.name
            shutil.copyfile(origfilename, file)

            # Open image
            svsImage = image_utils.open_svs_image_forced(file)
            imagesize = svsImage.get_level_dimension(config["detector.lod"])
            imagesize = (imagesize[1], imagesize[0], 3)

            if normalisation_filename:
                raise ValueError('Image normalisation has not been implemented when using disk based images')

            print('image filename: {}'.format(os.path.basename(origfilename)))
            print('image size: {}, {}'.format(imagesize[1], imagesize[0]))

            padding_x = inp_patch_size[1] - (imagesize[1] % inp_patch_size[1])
            padding_y = inp_patch_size[0] - (imagesize[0] % inp_patch_size[0])

            if config["segmentation.stain_transfer"]:
                if config["detector.lod"] < 2:
                    processinglod = 2
                else:
                    processinglod = config["detector.lod"]
                threshold = _calculate_mean(svsImage, processinglod)
            else:
                threshold = None

            h5pyfile = h5py.File(tmp_h5.name, 'w')
            segmentation = h5pyfile.create_dataset("segmentation", (
            imagesize[0] + padding_y, imagesize[1] + padding_x, number_of_classes), dtype=UNet.outputs[0].dtype.name)
            segmentation_count = h5pyfile.create_dataset("segmentation_count",
                                                         (imagesize[0] + padding_y, imagesize[1] + padding_x),
                                                         dtype=numpy.uint32)

            if config['detector.colour_mode'] == 'rgb':
                patch_batch = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], imagesize[2]),
                                          dtype=UNet.outputs[0].dtype.name)
                if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                    patch_batch_H = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], 1),
                                                dtype=UNet.outputs[0].dtype.name)
                    patch_batch_O = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], 1),
                                                dtype=UNet.outputs[0].dtype.name)
            else:
                patch_batch = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], 1),
                                          dtype=UNet.outputs[0].dtype.name)

            patch_coords = [None] * batch_size
            c = 0
            first_run = True
            t = []
            first_patch = True
            PATCH_SIZE = (4096, 4096)
            PATCH_SIZE = (int((((PATCH_SIZE[0] - inp_patch_size[0]) // stride) * stride) + inp_patch_size[0]),
                          int((((PATCH_SIZE[1] - inp_patch_size[1]) // stride) * stride) + inp_patch_size[1]))

            for y in range(0, imagesize[0], PATCH_SIZE[0] - inp_patch_size[0] + stride):
                for x in range(0, imagesize[1], PATCH_SIZE[1] - inp_patch_size[1] + stride):
                    sizey = min(PATCH_SIZE[0], imagesize[0] - y)
                    sizex = min(PATCH_SIZE[1], imagesize[1] - x)

                    # Threaded version
                    if not first_patch:
                        p.join()
                        imgpatch = p.imgpatch.astype(UNet.outputs[0].dtype.name)
                        patchx = p.x
                        patchy = p.y

                    p = ImagePatchReadThread(svsImage, x, y, config["detector.lod"], sizex, sizey)
                    p.start()

                    if not first_patch:
                        if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                            segmentation, segmentation_count, patch_batch, patch_batch_H, patch_batch_O, \
                                patch_coords, t, c, first_run = _process_patch(UNet, diff, imgpatch, inp_patch_size,
                                                                               otp_patch_size, stride, batch_size,
                                                                               patch_batch, patch_coords, patchx,
                                                                               patchy, standardise_patches,
                                                                               config['normalisation.normalise_patches'],
                                                                               mean, stddev, t, c, segmentation,
                                                                               segmentation_count, first_run,
                                                                               config['segmentation.stain_transfer'],
                                                                               threshold, config['general.datapath'],
                                                                               patch_batch_H, patch_batch_O)
                        else:
                            segmentation, segmentation_count, patch_batch, \
                                patch_coords, t, c, first_run = _process_patch(UNet, diff, imgpatch, inp_patch_size,
                                                                               otp_patch_size, stride, batch_size,
                                                                               patch_batch, patch_coords, patchx,
                                                                               patchy, standardise_patches,
                                                                               config['normalisation.normalise_patches'],
                                                                               mean, stddev, t, c, segmentation,
                                                                               segmentation_count, first_run,
                                                                               config['segmentation.stain_transfer'],
                                                                               threshold, config['general.datapath'])

                    first_patch = False
            # Threaded version
            if not first_patch:
                p.join()
                imgpatch = p.imgpatch.astype(UNet.outputs[0].dtype.name)
                patchx = p.x
                patchy = p.y

                if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                    segmentation, segmentation_count, patch_batch, patch_batch_H, patch_batch_O, \
                        patch_coords, t, c, first_run = _process_patch(UNet, diff, imgpatch, inp_patch_size,
                                                                       otp_patch_size, stride, batch_size, patch_batch, patch_coords,
                                                                       patchx, patchy, standardise_patches,
                                                                       config['normalisation.normalise_patches'],
                                                                       mean, stddev, t, c, segmentation,
                                                                       segmentation_count,
                                                                       first_run, config['segmentation.stain_transfer'],
                                                                       threshold, config['general.datapath'],
                                                                       patch_batch_H, patch_batch_O)
                else:
                    segmentation, segmentation_count, patch_batch, \
                        patch_coords, t, c, first_run = _process_patch(UNet, diff, imgpatch, inp_patch_size,
                                                                       otp_patch_size, stride, batch_size, patch_batch, patch_coords,
                                                                       patchx, patchy, standardise_patches,
                                                                       config['normalisation.normalise_patches'],
                                                                       mean, stddev, t, c, segmentation,
                                                                       segmentation_count,
                                                                       first_run, config['segmentation.stain_transfer'],
                                                                       threshold, config['general.datapath'])

            if c > 0:
                if config['transferlearning.finetune'] and config['transferlearning.finetunemode'] == "SSL" and config['transferlearning.pretrained_ssl_model'] == "hrcsco":
                    preds = UNet.predict([patch_batch_H[0:c, :, :, :], patch_batch_O[0:c, :, :, :]])
                else:
                    preds = UNet.predict(patch_batch[0:c, :, :, :])

                for i, coord in enumerate(patch_coords[0:c]):
                    segmentation[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                    coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1], :] += preds[i, :, :, :]
                    segmentation_count[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                    coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1]] += 1

            # Save the output prediction for each class
            for className in config['extraction.class_definitions'].keys():
                sys.stdout.write('\033[K')
                sys.stdout.write('Constructing {} segmentation\r'.format(className))
                label = config['extraction.class_definitions'][className][0]
                output_dir = os.path.join(img_output_path, str(className))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                image_file = memmap(tmp_tif.name,
                                    shape=(imagesize[0], imagesize[1]),
                                    dtype=numpy.uint8,
                                    bigsize=2 ** 31)
                for y in range(0, imagesize[0], PATCH_SIZE[1]):
                    for x in range(0, imagesize[1], PATCH_SIZE[0]):
                        y2 = min(y + PATCH_SIZE[1], imagesize[0])
                        x2 = min(x + PATCH_SIZE[0], imagesize[1])
                        patch = (numpy.divide(segmentation[y:y2, x:x2, label],
                                              (segmentation_count[y:y2, x:x2] + epsilon())) * 255).astype(numpy.uint8)
                        patch[segmentation_count[y:y2, x:x2] == 0] = 255
                        image_file[y:y2, x:x2] = patch
                image_file.flush()
                del image_file

                # the above memmap allows us to write the image without loading it all im memory in TIFF format,
                # however it is uncompressed. We now convert this file to PNG in a memory efficient way
                sys.stdout.write('\033[K')
                sys.stdout.write('Converting {} image format\r'.format(className))
                image_utils.convert_image(tmp_tif.name,
                                          os.path.join(output_dir,
                                                       os.path.splitext(os.path.basename(origfilename))[0] + '.png'))

            sys.stdout.write('\033[K')


def calculatemaxoutputsegmentations(imageList, segmentationpath, detectionpath, label, class_definitions):
    """

    calculatemaxoutputsegmentations create a thresholded image in which each pixel is the class predicted with highest
    probability is chosen based on the different segmentation results


    :param imageList: list of (string), the list of image names to analyse
    :param segmentationpath: (string) the path of the segmentations (where imageList exist)
    :param detectionpath: (string) the path of the detection folder (the output folder)
    :param label: (string) the label of the trained network
    :param class_definitions: dictionary of (string, tuple) that contains the classlabel (integer), extraction method
    (random or centred), and number of samples to extract for each class, a value of -1 means extract all possible
    patches
    """

    detectionpath = os.path.join(detectionpath, label, "maxoutput")
    if not os.path.exists(detectionpath):
        os.makedirs(detectionpath)

    segmentationpath = os.path.join(segmentationpath, label)

    for image in imageList:
        imageName = os.path.splitext(os.path.basename(image))[0]

        for i, c in enumerate(list(class_definitions)):
            segmentation = image_utils.read_image(os.path.join(segmentationpath, c, imageName + '.png'))
            if i == 0:
                prediction = segmentation.copy()
                detection = numpy.ones((segmentation.shape[0], segmentation.shape[1]), dtype=numpy.uint8) * \
                            class_definitions[c][0]
                mask = prediction == 255
            else:
                detection[segmentation > prediction] = class_definitions[c][0]
                prediction[segmentation > prediction] = segmentation[segmentation > prediction]
                mask = numpy.logical_and(mask, segmentation == 255)

        detection[mask] = 255
        image_utils.save_image(detection, os.path.join(detectionpath, imageName + '.png'))


def getpatientimagefilenames(datapath, patients):
    patientstrings = [patient.zfill(4) for patient in patients]

    images = []
    if patients is not None:
        for imagePath in sorted(glob.glob(os.path.join(datapath, "images", "*"))):
            imageName = imagePath.split(os.sep)[-1].split(".")[0]

            if any(a in imageName for a in patientstrings):
                images.append(imagePath)

    return images


def pretrained_csco_model_path(conf):
    return os.path.join(conf['general.homepath'], 'saved_models/SSL/Nephrectomy',
                        conf['transferlearning.pretrained_ssl_model'], conf['detector.segmentationmodel'],
                        conf['general.staincode'], "rep1/contrastive_learning/models/HO_encoder_model.best.hdf5")

def derived_parameters(conf, arguments):
    conf['detector.patchstrategy'] = arguments.patch_strategy
    conf['detector.percentN'] = arguments.percent_N

    if arguments.validation_data.lower() == "none":
        config['detector.validation_data'] = None
    else:
        config['detector.validation_data'] = arguments.validation_data
        
    if conf['detector.percentN'] == "percent_100":
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'patches')
    else:
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'separated_patches',
                                                       conf['detector.patchstrategy'], conf['detector.percentN'])

    if config['detector.validation_data'] is None:
        conf['detector.validationinputpath'] = None 
    elif conf['detector.validation_data'].lower() == "respective_splits":
        if conf['detector.percentN'] == "percent_100":
            conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'], 'patches')
        else:
            conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'], 'separated_patches',
                                                                conf['detector.patchstrategy'], conf['detector.percentN']) 
    elif conf['detector.validation_data'].lower() == "full":
        conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'], 'patches')
    
     
    if conf['detector.validation_data'] is None:
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'], 'without_validation_data')
    elif conf['detector.validation_data'].lower() == "respective_splits":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'], 'respective_splits_validation_data')
    elif conf['detector.validation_data'].lower() == "full":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'], 'full_validation_data')
    
    if conf['transferlearning.finetune']:
        conf['transferlearning.stain_separate'] = True
        conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model.lower()
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable
        
        if 'hrcsco' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.csco_model_path'] = pretrained_csco_model_path(conf)
        else:
            raise ValueError("Self-supervised learning based pretrained-models should be one of ['hrcsco']")

        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'],
                                                   conf['transferlearning.pretrained_ssl_model'])
    else:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'])

    conf['segmentation.segmentationpath'] = os.path.join(conf['detector.outputpath'],
                                                         conf['segmentation.segmentationpath'])
    conf['segmentation.detectionpath'] = os.path.join(conf['detector.outputpath'],
                                                      conf['segmentation.detectionpath'])
    return conf


def segmentation_exists(segmentation_path, stain):
    # print(f"Stain: {stain}")
    # print(glob.glob(os.path.join(segmentation_path, "*", f"*_{stain}.png")))
    if len(glob.glob(os.path.join(segmentation_path, "*", f"*_{stain}.png"))) == 12:
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a UNet model (default behaviour is to read the SVS images '
                                                 'defined by the test patients in the configuration file \'SSL*.cfg\').')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-l', '--label', type=str, help='the label of the model to be tested')
    parser.add_argument('-d', '--imagedir', type=str, help='the directory containing the SVS images to be segmented')
    parser.add_argument('-f', '--imagefile', type=str, help='the SVS image to be segmented')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
    parser.add_argument('-t', '--tmpdir', type=str, help='specify which GPU to use')

    # Adding parameters to finetune the UNet with pretrained Self Supervised Learning Models (SimCLR, Byol, CSCO, etc)
    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, default='CSCO')
    parser.add_argument('-pmt', '--pretrained_model_trainable', default=True, 
                        type=lambda x: str(x).lower() in ['true', '1', 'yes'])
    # Adding parameters to specify the data strategy
    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pn', '--percent_N', type=str, default='percent_1')
   
    parser.add_argument('-lr', '--LR', type=str, default="None")
    parser.add_argument('-lrd', '--LR_weightdecay', type=str, default="None")
    parser.add_argument('-rlrp', '--reduceLR_percentile', type=str, default="None")
    
    parser.add_argument('-vd', '--validation_data', type=str, default="None", help="none | respective_splits | full")
    

    start = datetime.datetime.now()
    args = parser.parse_args()

    # if args.gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
    #
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # print("\nGpu Growth Restriction Done...")

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    config = derived_parameters(config, arguments=args)
    
    if args.LR != 'None':
        config['detector.learn_rate'] = float(args.LR)

    if args.LR_weightdecay != 'None':
        config['detector.LR_weightdecay'] = float(args.LR_weightdecay)
    else:
        config['detector.LR_weightdecay'] = None
    
    if args.reduceLR_percentile != 'None' and config['transferlearning.finetune']:
        if config['detector.reducelr'] and config['transferlearning.pretrained_model_trainable']:
            config['detector.reduceLR_percentile'] = int(args.reduceLR_percentile)
            config['detector.reduceLR_patience'] = int((config['detector.reduceLR_percentile'] / 100) * config['detector.epochs'])

    if args.label:
        label = args.label

    if config['detector.segmentationmodel'] and not filepath_utils.validmodel(config['detector.segmentationmodel']):
        raise ValueError('The model should be one of ["unet", "deepresidualunet"]')

    print("#######################################################")
    print("##################### Apply UNET #####################")
    print("#######################################################")
    print("\nConfiguration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))

    filePath = filepath_utils.FilepathGenerator(config)

    normalisation_filename = None
    if config['normalisation.normalise_image']:
        normalisation_filename = os.path.join(config['detector.outputpath'], 'models', 'histogram_matching_stats.hdf5')

    modelfilename = f'{config["detector.segmentationmodel"]}_best.' + label + '.hdf5'

    if not args.imagedir and not args.imagefile:
        imageList = [f[0] for f in filePath.get_images_with_list_patients(patients=config['general.testPatients'])]
    elif args.imagefile:
        if not os.path.isfile(args.imagefile):
            raise ValueError("Input file %s does not exist" % args.imagefile)
        imageList = [args.imagefile]
    elif args.imagedir:
        if not os.path.isdir(args.imagedir):
            raise ValueError("Input directory %s does not exist" % args.imagedir)
        imageList = glob.glob(os.path.join(args.imagedir, "*.svs"))


    if os.path.isfile(os.path.join(config['detector.outputpath'],  'graphs', 'loss_history.' + label + '.png')):
        if not segmentation_exists(segmentation_path=os.path.join(config['segmentation.segmentationpath'], label),
                                   stain=args.imagedir.rsplit("/", 3)[1]):
            apply_model_from_disk(args.tmpdir, imageList, config, modelfilename, label, normalisation_filename)
            calculatemaxoutputsegmentations(imageList, config['segmentation.segmentationpath'],
                                            config['segmentation.detectionpath'], label,
                                           config['extraction.class_definitions'])
        else:
            print("Applying unet has already done")
    else:
        raise ValueError("Model does not finished the training properly...")

    print('\nTime Taken: {}'.format(datetime.datetime.now() - start))
