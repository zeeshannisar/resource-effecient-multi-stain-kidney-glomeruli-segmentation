import os
import argparse
import glob
import h5py
from utils import image_utils, config_utils, data_utils, filepath_utils,config_utils_stargan
import numpy
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import epsilon
from skimage.util import view_as_windows, pad, crop
#from unet.losses import weighted_categorical_crossentropy
import sys
from threading import Thread
import tensorflow as tff
from tifffile import memmap
import tempfile
import shutil
import warnings
from background_tissue_detection import _calculate_mean
import skimage.color
import staintools
import random
import os
from utils.select_gpu import  pick_gpu_lowest_memory
from cycgan import CycleGAN_models
import numpy as np
from unet import unet_models
import tensorflow as tf
import tensorflow_addons as tfa
from unet.unet_models import ConvDownModule
import datetime
import json


# todo: openslide read region and read whole image give different results for the same image and region! So process from disk and process from memory will give slightly different resutls...

#class NetworkPredictThread(Thread):
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


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.shape[0]
    #print(batch_size)
    out = np.zeros(shape=(batch_size, dim))
    # print(out.shape)
    # print(labels)
    out[np.arange(batch_size), labels.astype(np.int)] = 1
    # print(out)
    return out.astype(np.int)


def _process_patch(UNet, diff, imgpatch, inp_patch_size, otp_patch_size, stride, batch_size, patch_batch, patch_coords,
                   patchx, patchy, standardise_patches, normalise_patches, mean, stddev, t, c, segmentation,
                   segmentation_count, first_run, staintransfer=False, threshold=None, data_dir=None, transform_fun=None):

    for i in range(0, imgpatch.shape[0] - (inp_patch_size[0] - 1), stride):
        for j in range(0, imgpatch.shape[1] - (inp_patch_size[1] - 1), stride):

            patch = imgpatch[i:i + inp_patch_size[0], j:j + inp_patch_size[1]]

            if staintransfer:
                if transform_fun is not None:
                    patch = (patch / 127.5) - 1  # [-1,1]
                    patch = transform_fun.predict_on_batch(patch[numpy.newaxis, ...])
                    patch = (patch + 1) * 127.5  # [0-255]
                    # import uuid
                    # image_utils.save_image(patch[0, :, :, :], '../apply_on_converted_images/' + str(uuid.uuid4()) + '.png')

            patch = image_utils.image_colour_convert(patch, config['detector.colour_mode'])

            if standardise_patches:
                patch = data_utils.standardise_sample(patch)

            if normalise_patches:
                patch = data_utils.normalise_sample(patch, mean, stddev)

            patch_batch[c, :, :, :] = patch

            patch_coords[c] = (patchy + i, patchx + j)

            c += 1

            if c == batch_size:

                # Non threaded version
                sys.stdout.write('x: {:^6}, y: {:^6}\r'.format(patchx + j, patchy + i))
                preds = UNet.predict(patch_batch)

                for k, coord in enumerate(patch_coords):
                    segmentation[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                    coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1], :] += preds[k, :, :, :]
                    segmentation_count[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0],
                    coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1]] += 1
                t = None

                # Threaded version
                #if not first_run:
                #    t.join()
                #    preds = t.prediction
                #    prev_patch_coords = t.patch_coords

                #    sys.stdout.write('x: {:^5}, y: {:^5}\r'.format(patchx + j, patchy + i))

                #    for k, coord in enumerate(prev_patch_coords):
                #        segmentation[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0], coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1], :] += preds[k, :, :, :]
                #        segmentation_count[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0], coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1]] += 1

                #t = NetworkPredictThread(UNet, patch_batch.copy(), patch_coords.copy())
                #t.start()

                first_run = False
                c = 0

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
    #     warnings.warn("This function writes large amounts of temporary data to disk, there is less then 70GB free on " + tempfile.gettempdir() + ", do you want to continue?")
    #     cont = ''
    #     while cont not in ['yes', 'no']:
    #         cont = input("yes/no > ")
    #     if not cont == "yes":
    #         return

    modelpath  =          config["detector.outputpath"]
    stride     =          config['segmentation.stride']
    batch_size =          config['segmentation.batch_size']
    outputpath =          config['segmentation.segmentationpath']
    standardise_patches = config['normalisation.standardise_patches']

    if normalisation_filename and config['normalisation.normalise_within_tissue'] and sys.version_info[0] != 2:
        # Only if normalising within tissue
        raise RuntimeError("Only Python 2 is supported")

    # UNet, _, _ = unet_models.build_UNet((508, 508, 3), 3,
    #                                          depth=config['detector.network_depth'],
    #                                          filter_factor_offset=config['detector.filter_factor_offset'],
    #                                          initialiser=config['detector.weightinit'],
    #                                          padding=config['detector.padding'],
    #                                          modifiedarch=config['detector.modifiedarch'],
    #                                          batchnormalisation=config['detector.batchnormalisation'],
    #                                          k_size=config['detector.kernel_size'],
    #                                          dropout=config['detector.dropout'],
    #                                          learnupscale=config['detector.learnupscale'],
    #                                          learncolour=False)
    #
    # UNet.load_weights(os.path.join(modelpath, 'models', modelfilename))

    if config['detector.LR_weightdecay'] is not None:
        print("tfa.optimizers.AdamW")
        optimiser = tfa.optimizers.AdamW(learning_rate=config['detector.learn_rate'], beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-08, weight_decay=config['detector.LR_weightdecay'], name='AdamW')

        if config['transferlearning.finetune']:
            if 'byol' in config['transferlearning.pretrained_ssl_model']:
                custom_objects = {'AdamW': optimiser, 'ConvDownModule': ConvDownModule}
            else:
                custom_objects = {'AdamW': optimiser}
    else:
        print("tf.keras.optimizers.Adam")
        if config['transferlearning.finetune']:
            if 'byol' in config['transferlearning.pretrained_ssl_model']:
                custom_objects = {'ConvDownModule': ConvDownModule}
            else:
                custom_objects = None
        else:
            custom_objects = None
            
            
    UNet = load_model(os.path.join(modelpath, 'models', modelfilename), custom_objects=custom_objects)

    UNet.make_predict_function()  # Required for executing predict in a separate thread

    number_of_classes = UNet.outputs[0].shape[-1]

    if config['normalisation.normalise_patches']:
        # Read normalisation statistics
        statsfilename = os.path.join(modelpath, 'models', "normalisation_stats." + modeltimestamp + ".hdf5")
        with h5py.File(statsfilename, "r") as f:
            mean = f["stats"][0]
            stddev = f["stats"][1]
    else:
        mean = None
        stddev = None

    print(f"Mean: {mean} --- Stddev: {stddev}")

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
            segmentation = h5pyfile.create_dataset("segmentation", (imagesize[0] + padding_y, imagesize[1] + padding_x, number_of_classes), dtype=UNet.outputs[0].dtype.name)
            segmentation_count = h5pyfile.create_dataset("segmentation_count", (imagesize[0] + padding_y, imagesize[1] + padding_x), dtype=numpy.uint32)

            if config['detector.colour_mode'] == 'rgb':
                patch_batch = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], imagesize[2]), dtype=UNet.outputs[0].dtype.name)
            else:
                patch_batch = numpy.zeros(shape=(batch_size, inp_patch_size[0], inp_patch_size[1], 1), dtype=UNet.outputs[0].dtype.name)

            patch_coords = [None] * batch_size
            c = 0

            first_run = True
            t = []
            first_patch = True
            PATCH_SIZE = (4096, 4096)
            PATCH_SIZE = (int((((PATCH_SIZE[0] - inp_patch_size[0]) // stride) * stride) + inp_patch_size[0]),
                                  int((((PATCH_SIZE[1] - inp_patch_size[1]) // stride) * stride) + inp_patch_size[1]))

            for y in range(0, imagesize[0], PATCH_SIZE[0]-inp_patch_size[0]+stride):
                for x in range(0, imagesize[1], PATCH_SIZE[1]-inp_patch_size[1]+stride):
                    sizey = min(PATCH_SIZE[0], imagesize[0] - y)
                    sizex = min(PATCH_SIZE[1], imagesize[1] - x)

                    # Non threaded version
                    #imgpatch = svsImage.read_region((x, y), config["detector.lod"], (sizex, sizey))
                    #patchx = x
                    #patchy = y

                    # Threaded version
                    if not first_patch:
                        p.join()
                        imgpatch = p.imgpatch.astype(UNet.outputs[0].dtype.name)
                        patchx = p.x
                        patchy = p.y

                    p = ImagePatchReadThread(svsImage, x, y, config["detector.lod"], sizex, sizey)
                    p.start()

                    if not first_patch:
                    #if True:
                        segmentation, segmentation_count, patch_batch, patch_coords, t, c, first_run = _process_patch(UNet,
                                                                                                            diff,
                                                                                                            imgpatch,
                                                                                                            inp_patch_size,
                                                                                                            otp_patch_size,
                                                                                                            stride,
                                                                                                            batch_size,
                                                                                                            patch_batch,
                                                                                                            patch_coords,
                                                                                                            patchx,
                                                                                                            patchy,
                                                                                                            standardise_patches,
                                                                                                            config['normalisation.normalise_patches'],
                                                                                                            mean,
                                                                                                            stddev, t, c,
                                                                                                            segmentation,
                                                                                                            segmentation_count,
                                                                                                            first_run,
                                                                                                            config['segmentation.stain_transfer'],
                                                                                                            threshold,
                                                                                                            config['general.datapath'])

                    first_patch = False

            # Threaded version
            if not first_patch:
                p.join()
                imgpatch = p.imgpatch.astype(UNet.outputs[0].dtype.name)
                patchx = p.x
                patchy = p.y

                segmentation, segmentation_count, patch_batch, patch_coords, t, c, first_run = _process_patch(UNet,
                                                                                                    diff,
                                                                                                    imgpatch,
                                                                                                    inp_patch_size,
                                                                                                    otp_patch_size,
                                                                                                    stride,
                                                                                                    batch_size,
                                                                                                    patch_batch,
                                                                                                    patch_coords,
                                                                                                    patchx,
                                                                                                    patchy,
                                                                                                    standardise_patches,
                                                                                                    config['normalisation.normalise_patches'],
                                                                                                    mean,
                                                                                                    stddev, t, c,
                                                                                                    segmentation,
                                                                                                    segmentation_count,
                                                                                                    first_run,
                                                                                                    config['segmentation.stain_transfer'],
                                                                                                    threshold,
                                                                                                    config['general.datapath'])

            if c > 0:
                preds = UNet.predict(patch_batch[0:c, :, :, :])
                for i, coord in enumerate(patch_coords[0:c]):
                    segmentation[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0], coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1], :] += preds[i, :, :, :]
                    segmentation_count[coord[0] + diff[0]:coord[0] + diff[0] + otp_patch_size[0], coord[1] + diff[1]:coord[1] + diff[1] + otp_patch_size[1]] += 1

            #numpy.save('new_tool_box_segmentation.npy',segmentation)
            #numpy.save('new_tool_box_segmentation_count.npy',segmentation_count)
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
                                              (segmentation_count[y:y2,
                                               x:x2] + epsilon())) * 255).astype(numpy.uint8)
                        patch[segmentation_count[y:y2, x:x2] == 0] = 255
                        image_file[y:y2, x:x2] = patch
                image_file.flush()
                del image_file

                # the above memmap allows us to write the image without loading it all im memory in TIFF format,
                # however it is uncompressed. We now convert this file to PNG in a memory efficient way
                sys.stdout.write('\033[K')
                sys.stdout.write('Converting {} image format\r'.format(className))
                image_utils.convert_image(tmp_tif.name,
                                          os.path.join(output_dir, os.path.splitext(os.path.basename(origfilename))[0] + '.png'))

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
                detection = numpy.ones((segmentation.shape[0], segmentation.shape[1]), dtype=numpy.uint8) * class_definitions[c][0]
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


def pretrained_simclr_model_path(conf):
    return os.path.join(conf['general.homepath'], 'code/improve_kidney_glomeruli_segmentation/pre_training',
                        'ssl_pretrained_models/simclr/simclr_unet_encoder.h5')

def pretrained_byol_model_path(conf):
    return os.path.join(conf['general.homepath'], 'code/improve_kidney_glomeruli_segmentation/pre_training',
                        'ssl_pretrained_models/byol/byol_unet_encoder.h5')

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
                                              conf['detector.percentN'], conf['general.staincode'])
    

    if conf['transferlearning.finetune']:
        conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model.lower()
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable

        if 'simclr' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.simclr_model_path'] = pretrained_simclr_model_path(conf)
        elif 'byol' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.byol_model_path'] = pretrained_byol_model_path(conf)
        else:
            raise ValueError("Self-supervised learning based pretrained-models should be one of ['simclr', 'byol', 'hrcsco']")

    if conf['transferlearning.finetune']:
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
    parser = argparse.ArgumentParser(description='Test a UNet model (default behaviour is to read the SVS images defined by the test patients in the configuration file \'code.cfg\').')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-l', '--label', type=str, help='the label of the model to be tested')
    parser.add_argument('-d', '--imagedir', type=str, help='the directory containing the SVS images to be segmented')
    parser.add_argument('-f', '--imagefile', type=str, help='the SVS image to be segmented')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
    parser.add_argument('-t', '--tmpdir', type=str, help='specify which GPU to use')
   
    # Adding parameters to finetune the UNet with pretrained Self Supervised Learning Models (SimCLR, Byol, etc)
    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, help='simclr | byol | none')
    parser.add_argument('-pmt', '--pretrained_model_trainable', default=False,
                        type=lambda x: str(x).lower() in ['true', '1', 'yes'], help='if finetune: True | if fixedfeatures: False')

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
    # print("Selected GPU : " + os.environ["CUDA_VISIBLE_DEVICES"])
    # print("\n")

    # # Allow memory growth
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # print("\nGpu Growth Restriction Done...")

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    config = derived_parameters(config, arguments=args)

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
    else:
        raise ValueError('The label is a must argument to pass.')

    # Read configuration
    # Models are trained on GENCI so no need to read the configuration files here as it has paths according to
    # with open(os.path.join(config['detector.outputpath'], 'SSL.' + label + '.json'), 'r') as f:
    #     config = json.load(f)

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
        print('Apply images from '+args.imagedir)
        if not os.path.isdir(args.imagedir):
            raise ValueError("Input directory %s does not exist" % args.imagedir)
        imageList = glob.glob(os.path.join(args.imagedir, "*.svs"))

    print(os.path.isfile(os.path.join(config['detector.outputpath'],  'graphs', 'loss_history.' + label + '.png')))
    if os.path.isfile(os.path.join(config['detector.outputpath'],  'graphs', 'loss_history.' + label + '.png')):
        apply_model_from_disk(args.tmpdir, imageList, config, modelfilename, label, normalisation_filename)
        calculatemaxoutputsegmentations(imageList, config['segmentation.segmentationpath'],
                                        config['segmentation.detectionpath'], label,
                                        config['extraction.class_definitions'])
        # if not segmentation_exists(segmentation_path=os.path.join(config['segmentation.segmentationpath'], label),
        #                            stain=args.imagedir.rsplit("/", 3)[1]):
        #     apply_model_from_disk(args.tmpdir, imageList, config, modelfilename, label, normalisation_filename)
        #     calculatemaxoutputsegmentations(imageList, config['segmentation.segmentationpath'],
        #                                     config['segmentation.detectionpath'], label,
        #                                     config['extraction.class_definitions'])
        # else:
        #     print("Applying unet has already done")
    else:
        raise ValueError("Model does not finished the training properly...")

    print('\nTime Taken: {}'.format(datetime.datetime.now() - start))
