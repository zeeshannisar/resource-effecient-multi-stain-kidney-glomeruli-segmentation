# -*- coding: utf-8 -*-
#  Odyssee Merveille 08/11/17
"""
extract_patches.py:
"""
from utils import config_utils, filepath_utils
import numpy
import argparse
import shutil
import warnings
import os
import copy

from patch_extractors.uniform_extractor import UniformExtractor
from patch_extractors.centroid_extractor import CentroidExtractor
from patch_extractors.random_extractor import RandomExtractMethod


# todo: Image normalisation not working due to reading patches from disk instead of loading full image

def extract_patches(filePath, output_path, lod, patch_size, patients, class_definitions, h5py_norm_filename, normalise_within_tissue, uniform_overlap=0):
    """
    extract_patches: extract negative and positive patches from images.
    :param filePath: (FilePathGenerator) the file path used for the creation of path
    :param output_path: (string) the directory in which to write the results
    :param lod: (int) the level of detail required
    :param patch_size: (int) the size of the patches to be extracted
    :param patients: (list of int) the list of patients from which to extract patc  hes
    :param nb_patches_per_class: array of (int, int) dictionary that contains the number of patches to extract for each
    positive class, a value of -1 means extract all possible patches
    :param nb_neg: (int) the number of negative (i.e. parts of the image that do no makeup a positive class) patches to
    extract, a value of -1 means extract all possible patches
    :param h5py_norm_filename: (string) the path of the file h5py_norm used for image normalisation
    :param normalise_within_tissue: (boolean) only normalise within tissue or not
    """

    if not patients:
        warnings.warn('No patients requested.')
        return

    class_definitions = copy.deepcopy(class_definitions)

    # Check that at least one of the images exists
    existing_patients = filePath.getpatientlist()

    existing_requested_patients = list(set(patients) & set(existing_patients))

    if len(existing_requested_patients) < len(patients):
        warnings.warn('No images and GT for patients %s found (%i valid patients found).'
                      % (', '.join(set(patients) - set(existing_requested_patients)), len(existing_requested_patients)))

    if existing_requested_patients:
        classNames = list(class_definitions.keys())

        extractors = {}
        for className in classNames:
            extraction_method = class_definitions[className][1]
            if extraction_method == 'random':
                extractors[className] = RandomExtractMethod(filePath, output_path, lod, patch_size, h5py_norm_filename, normalise_within_tissue)
            elif extraction_method == 'centred':
                extractors[className] = CentroidExtractor(filePath, output_path, lod, patch_size, h5py_norm_filename, normalise_within_tissue)
            elif extraction_method == 'uniform':
                extractors[className] = UniformExtractor(filePath, output_path, lod, patch_size, h5py_norm_filename, normalise_within_tissue, overlap=uniform_overlap)
            else:
                raise ValueError('Unknown extraction method, the options are: random, centred, or uniform')

        #  If the user requested all the patches of at least one class, calculate their number
        listAllPatchIndices = [k for k, v in class_definitions.items() if v[-1] == -1]
        if listAllPatchIndices:
            if not all([class_definitions[className][1] == 'centred' or class_definitions[className][1] == 'uniform' for className in listAllPatchIndices]):
                warnings.warn('Extract all patches is intended for objects extracted using the centred method (this is just a warning)')
            for className in listAllPatchIndices:
                patch_number = extractors[className].compute_nb_patches(patients, className, class_definitions[className][0])
                if patch_number == 0:
                    warnings.warn('No patches found for class %s' % className)
                class_definitions[className][-1] = patch_number

        # Calculate sample numbers for relatively defined classes
        for className in classNames:
            if class_definitions[className][2] == 'relative':
                target = class_definitions[className][3]
                fraction = class_definitions[className][-1]
                number = int(round(float(class_definitions[target][-1]) * fraction))
            else:
                number = class_definitions[className][-1]

            class_definitions[className] = [class_definitions[className][0], class_definitions[className][1], number]

        for className in classNames:
            if class_definitions[className][2] < 0:
                raise ValueError('Invalid number of patches for %s (%d), possible values are -1 (all patches) and > 0' % (className, class_definitions[className][2]))
            elif not class_definitions[className][2] == 0:
                extractors[className].extract_patches(patients, className, class_definitions[className][0], class_definitions[className][2])

        #print("\n \n")
        #print("Number of patches per class:")
        #for i, className in enumerate(classNames):
        #    print("    " + className + ": " + str(patch_extracted[i])+ " (on  " + str(class_definitions[className][2]) +
        #          " requested)")
        #print("\n \n")


def extract_unsupervised_patches(filePath, output_path, lod, patch_size, patients, number_of_patches, h5py_norm_filename, normalise_within_tissue, uniform_overlap=0):
    """
    extract_unsupervised_patches: extract tissue patches from images.
    :param filePath: (FilePathGenerator) the file path used for the creation of path
    :param output_path: (string) the directory in which to write the results
    :param lod: (int) the level of detail required
    :param patch_size: (int) the size of the patches to be extracted
    :param patients: (list of int) the list of patients from which to extract patc  hes
    :param nb_patches_per_class: array of (int, int) dictionary that contains the number of patches to extract for each
    positive class, a value of -1 means extract all possible patches
    :param nb_neg: (int) the number of negative (i.e. parts of the image that do no makeup a positive class) patches to
    extract, a value of -1 means extract all possible patches
    :param h5py_norm_filename: (string) the path of the file h5py_norm used for image normalisation
    :param normalise_within_tissue: (boolean) only normalise within tissue or not
    """

    if not patients:
        warnings.warn('No patients requested.')
        return

    # Check that at least one of the images exists
    existing_patients = filePath.getpatientlist()

    existing_requested_patients = list(set(patients) & set(existing_patients))

    if len(existing_requested_patients) < len(patients):
        warnings.warn('No images and GT for patients %s found (%i valid patients found).'
                      % (', '.join(set(patients) - set(existing_requested_patients)), len(existing_requested_patients)))

    if existing_requested_patients:
        extractor = RandomExtractMethod(filePath, output_path, lod, patch_size, h5py_norm_filename, normalise_within_tissue)

        if number_of_patches < 0:
            raise ValueError('Invalid number of tissue patches, possible values are > 0')
        elif not number_of_patches == 0:
            extractor.extract_tissue_patches(patients, 'tissue', number_of_patches)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    filePath = filepath_utils.FilepathGenerator(config)

    subdirs = ['train', 'validation', 'test']

    h5py_norm_filename = None
    if config['normalisation.normalise_image']:

        raise ValueError('Image normalisation has not been implemented when using disk based images')

        patient_selected = numpy.random.choice(config['general.trainPatients'])

        image_selected = filePath.get_images(patient=patient_selected)[0]

        prototype_image = filePath.get_mask_for_image(image_selected[1], 'tissue', config['general.lod'])

        h5py_norm_filename = os.path.join(config['extraction.extractbasepath'], 'histogram_matching_stats.hdf5')

        if config['normalisation.normalise_within_tissue']:
            tissue_file = prototype_image[0]

            image_utils.write_normalisation_data(image_selected[0], config['normalisation.normalise_within_tissue'],
                                                 config['general.lod'],
                                                 h5py_norm_filename,
                                                 tissue_file)
        else:
            image_utils.write_normalisation_data(image_selected[0], config['normalisation.normalise_within_tissue'],
                                                 config['general.lod'], h5py_norm_filename)

    shutil.rmtree(config['extraction.patchpath'], ignore_errors=True)

    print("Patients used for training (%d): %s" % (len(config['general.trainPatients']), ', '.join(config['general.trainPatients'])))
    print("Patients used for validation (%d): %s" % (len(config['general.validationPatients']), ', '.join(config['general.validationPatients'])))
    print("Patients used for testing (%d): %s" % (len(config['general.testPatients']), ', '.join(config['general.testPatients'])))
    #  Run the patch extraction

    for patients, subdir in zip([config['general.trainPatients'], config['general.validationPatients'], config['general.testPatients']], subdirs):
        outputdir = os.path.join(config['extraction.patchpath'], subdir)
        extract_patches(filePath,
                        outputdir,
                        config['general.lod'],
                        config['extraction.patch_size'],
                        patients,
                        config['extraction.class_definitions'],
                        h5py_norm_filename,
                        config['normalisation.normalise_within_tissue'],
                        uniform_overlap=config['extractor.uniform_overlap'])

    outputdir = os.path.join(config['extraction.patchpath'], 'colour')
    extract_unsupervised_patches(filePath,
                                 outputdir,
                                 config['general.lod'],
                                 config['extraction.patch_size'],
                                 config['general.colourPatients'],
                                 config['general.colourPatches'],
                                 h5py_norm_filename,
                                 config['normalisation.normalise_within_tissue'],
                                 uniform_overlap=config['extractor.uniform_overlap'])
