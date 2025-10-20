import os
import argparse
import glob
from random import shuffle
from utils import config_utils
import shutil
import json


# python3 combine_stain_trainingdata.py.py 02,03,16,32,39 -c sysmifta_Nx_02_03_16_32_39_new.cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('stains', type=str, help='comma separated list of stain codes to combine')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    staincodes = args.stains.split(',')

    outputstaincode = '_'.join(staincodes)

    shutil.rmtree(os.path.join(config['general.datapath'], outputstaincode, 'downsampledpatches'), ignore_errors=True)

    # Count number of patches for each staining
    for staincode in staincodes:

        for t in ['train', 'validation']:
            data_path = os.path.join(config['general.datapath'], staincode, 'downsampledpatches', t)
            inputlabelfile = os.path.join(data_path, 'class_labels.json')

            for classname in config['extraction.class_definitions'].keys():

                imgOutputDir = os.path.join(config['general.datapath'], outputstaincode, 'downsampledpatches', t,
                                            'images', classname)
                if not os.path.exists(imgOutputDir):
                    os.makedirs(imgOutputDir)
                print(imgOutputDir)
                gtOutputDir = os.path.join(config['general.datapath'], outputstaincode, 'downsampledpatches', t, 'gts',
                                           classname)
                if not os.path.exists(gtOutputDir):
                    os.makedirs(gtOutputDir)

                # TODO: Validate that the classes are the same (from meta files)
                # TODO: Redo number calculations with relative counts

                outputlabelfile = os.path.join(config['general.datapath'], outputstaincode, 'downsampledpatches', t, 'class_labels.json')
                if os.path.isfile(outputlabelfile):
                    # check that gt labels are the same as existing
                    with open(inputlabelfile, 'r') as fp:
                        imageLabels = json.load(fp)
                    with open(outputlabelfile, 'r') as fp:
                        patchLabels = json.load(fp)
                    if not imageLabels == patchLabels:
                        raise ValueError("Image labels differ from existing patch labels")
                else:
                    # copy gt label file
                    shutil.copyfile(inputlabelfile, outputlabelfile)

                if config['extraction.class_definitions'][classname][2] == 'absolute':
                    sample_number = config['extraction.class_definitions'][classname][3]
                else:
                    sample_number = config['extraction.class_definitions'][classname][4]
                sample_number = -1

                if sample_number > 0 or sample_number == -1:

                    nbSamples = sample_number // len(staincodes)

                    imgInputDir = os.path.join(data_path, 'images', classname)
                    imageList = os.listdir(imgInputDir)
                    gtInputDir = os.path.join(data_path, 'gts', classname)

                    if sample_number == -1:
                        nbSamples = len(imageList)

                    shuffle(imageList)
                    imageList = imageList[:nbSamples]

                    for image in imageList:
                        shutil.copyfile(os.path.join(imgInputDir, image), os.path.join(imgOutputDir, image))
                        shutil.copyfile(os.path.join(gtInputDir, image), os.path.join(gtOutputDir, image))
