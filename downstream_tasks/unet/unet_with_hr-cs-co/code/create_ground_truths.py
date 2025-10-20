"""
    create_groud_truths.py: used to create multi-class masks and groundtruths
"""

from utils import image_utils, config_utils, filepath_utils
import argparse
import os
import numpy
import pyvips
import json


'''
    Compute additional class by merging some existing classes
    INPUT:
            - maskDir: image directory containing all the masks
            - objectLabel: name of new class to be created
            - constituentObjects: list of all the labels of the classes to be combined (ex: [healthy, dead, sclerotic])
            - staincode: stain code
            - lod: lod at which the masks are saved.
'''


def combine_classes(filePath, objectLabel, constituentObjects, lod):
    """

    combine_classes: compute an additional class by merging some of the existing classes, a new folder is created in
    "maskpath"

    :param filePath: (FilepathGenerator) object used for finding the project's masks
    :param objectLabel: (string) the name of new class to be created
    :param constituentObjects: list of (string) list of all the labels of the classes to be combined (ex: [healthy, dead, sclerotic])
    :param lod: (int) the level of detail used
    """

    listImageName = filePath.find_filenames_with_specific_lod(constituentObjects[0], lod)

    for imageName in listImageName:
        print("Image: %s" % imageName)

        read = False
        for label in constituentObjects:

            maskPath = filePath.generate_maskpath(imageName, label, lod)

            print("Mask: %s" % maskPath)

            if os.path.isfile(maskPath):
                if read:
                    img = img.bandjoin(pyvips.Image.new_from_file(maskPath, access='sequential'))
                else:
                    img = pyvips.Image.new_from_file(maskPath, access='sequential')
                    read = True

        img = img.bandor()
        if read:
            maskOutputPath = filePath.generate_maskpath(imageName, objectLabel, lod)
            img.write_to_file(maskOutputPath)


def create_ground_truth(filePath, class_definitions, lod):
    """

    create_ground_truth: create the ground truth for every image. A ground truth is all the combined masks for one image

    :param filePath: (FilepathGenerator) object used for finding the masks of the project
    :param class_definitions: dictionary of (string, tuple) that contains the classlabel (integer), extraction method
    (random or centred), and number of samples to extract for each class, a value of -1 means extract all possible
    patches
    :param lod: (int) the level of detail used in the project
    """

    negindex = list(class_definitions).index("negative")
    if negindex == 0:
        index = 1
    else:
        index = 0
    listImageName = filePath.find_filenames_with_specific_lod(list(class_definitions)[index], lod)

    order = []
    for className in class_definitions.keys():
        order.append(class_definitions[className][0])
    indexes = numpy.argsort(order)
    classNames = [list(class_definitions)[i] for i in indexes]

    for imageName in listImageName:
        print("Image: %s" % imageName)

        read = False

        classlabels = {'negative': 0}
        for className in classNames:
            if not className == 'negative':
                class_label = class_definitions[className][0]
                
                classlabels[className] = class_label

                maskPath = filePath.generate_maskpath(imageName, className, lod)

                print("Mask: %s" % maskPath)

                if os.path.isfile(maskPath):
                    if read:
                        tmp_mask = image_utils.read_image(maskPath)
                        ground_truth[tmp_mask > 0] = class_label
                    else:
                        ground_truth = image_utils.read_image(maskPath).astype(numpy.uint8)
                        ground_truth[ground_truth > 0] = class_label
                        read = True

        if read:
            maskOutputPath = filePath.generate_groundtruthpath(imageName, lod)
            labelsOutputPath = filePath.generate_groundtruthlabelspath(imageName, lod)

            image_utils.save_image(ground_truth, maskOutputPath)
            with open(labelsOutputPath, 'w') as fp:
                json.dump(classlabels, fp, sort_keys=True, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration filename to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    filePath = filepath_utils.FilepathGenerator(config)

    for key in config['extraction.class_merges']:

        combine_classes(filePath, key, config['extraction.class_merges'][key], config['general.lod'])

        if config['general.lod'] != config['detector.lod']:
            combine_classes(filePath, key, config['extraction.class_merges'][key], config['detector.lod'])

    create_ground_truth(filePath, config['extraction.class_definitions'], config['general.lod'])

    if config['general.lod'] != config['detector.lod']:
        create_ground_truth(filePath, config['extraction.class_definitions'], config['detector.lod'])
