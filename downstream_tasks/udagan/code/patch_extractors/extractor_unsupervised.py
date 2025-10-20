"""
    extractor.py : file that contains the abstract class for every patches extractor
"""
import skimage.measure
import scipy.ndimage
import os
import numpy
import abc
import warnings
import sys
import os.path
from shutil import copyfile
import json

from utils import image_utils


class ExtractMethod(object):
    """
    ExtractMethod : abstract class used for the extraction of the patches
    """

    def __init__(self, file_path, output_folder, lod, patch_size, h5py_norm_filename, normalise_within_tissue):
        """
            constructor of the Extract method

        :param file_path: (FilepathGenerator) controller of the file path contained in the project
        :param output_folder: (string) the folder where are sent the result obtained
        :param lod: (int) the lod used for the image
        :param patch_size: (int) the requested size of the patch
        :param h5py_norm_filename: (string) the h5py path used for the normalization operation
        :param normalise_within_tissue: (bool) activate the normalization based on the elements contained in the tissues
        """
        self.file_path = file_path
        self.output_folder = output_folder
        self.lod = lod
        self.patch_size = patch_size
        self.h5py_norm_filename = h5py_norm_filename
        self.normalise_within_tissue = normalise_within_tissue

    def extract_tissue_patches(self, patients, className, nbPatches, debug=False):
        """
            extract_patches: for a list of patients asked : extract a certain number of patches corresponding on the
            requested labels

        :param patients: (list of string) the list of patients that will be extracted , every valid images that cotains
        these patients will be extracted
        :param className: (string) the name of the class used
        :param classLabel: (int) the number of the class used
        :param nbPatches: (int) the number of patches to extract, It is the Maximum requested ,
        depending on the algorithm used, the number of extracted patches can be lower
        :param debug: (bool) if true , the procedure makes extra operation, it draw a map from where the patches has been
        extracted
        :return : (int) the number of patches extracted for the class
        """

        print("Creating %d %s patches" % (nbPatches, className))

        # Get the mask path for each image
        listImagePaths, listgtPaths = self.file_path.get_patient_images_with_gts(self.lod, patients=patients)
        listNames = [os.path.splitext(os.path.basename(ip))[0] for ip in listImagePaths]

        # check that gt labels match extractor definitions
        maskPaths = []
        for imageName, imagepath in zip(listNames, listImagePaths):
            maskPaths.append(self.file_path.get_mask_for_image(os.path.join(imagepath, imageName)), 'tissue', self.lod)

        # compute the number of patches requested
        nbPatchPerImage = self.compute_nb_patches_per_image(nbPatches, maskPaths, 1)

        if numpy.any(nbPatchPerImage):
            self.create_folders(className, debug)

        for imageName, imagepath, maskpath, nb in zip(listNames, listImagePaths, maskPaths, nbPatchPerImage):
            print("Image: %s" % imageName)

            mask = image_utils.read_image(maskpath)

            if self.h5py_norm_filename:
                tissuePath = self.file_path.get_mask_for_image(imageName, 'tissue', self.lod)[0]
                image = self.normalise_image(image, tissuePath)

            patch_coords = self.get_patch_coords(mask, 1, nb)
            if len(patch_coords) < nb and debug:
                raise ValueError("Not enough samples for class %s for the number of patches requested in %s (found %s, requested %s) taking the lower value" % (className, imageName, numberPatchesCreated, nb))

            self.create_patches(imagepath, patch_coords, className, imageName, debug)

    @abc.abstractmethod
    def compute_nb_patches_per_image(self, nbPatch, listgts, classLabel):
        """
        compute_nb_patches_per_image : compute the number of patches for each images based on the number of patches
        wanted

        :param nbPatch: (int) the number maximum of patches requested
        :param listImage: (list of numpy.array) the images used in the extraction
        :param listgts: (list of numpy.array) the groundtruth used in the extraction, should be in the same order that
        the listImage
        :param classLabel: (int) the label used on  groundtruth
        :return: (list of int) the list of number of patches to extract for each images
        """
        raise NotImplemented()

    def compute_nb_patches(self, patients, className, classLabel):
        """
        compute_nb_patches_per_image : compute the number of patches for each images based on the number of patches
        wanted

        :param nbPatch: (int) the number maximum of patches requested
        :param listImage: (list of numpy.array) the images used in the extraction
        :param listgts: (list of numpy.array) the groundtruth used in the extraction, should be in the same order that
        the listImage
        :param classLabel: (int) the label used on  groundtruth
        :return: (list of int) the list of number of patches to extract for each images
        """
        listImagePaths, listgtPaths = self.file_path.get_patient_images_with_gts(self.lod, patients=patients)

        listNames = [os.path.splitext(os.path.basename(ip))[0] for ip in listImagePaths]

        for imageName in listNames:
            # check that gt labels mlabel_valid_regionsatch extractor definitions
            imageLabelPath = self.file_path.generate_groundtruthlabelspath(imageName, self.lod)
            with open(imageLabelPath, 'r') as fp:
                imageLabels = json.load(fp)
            if not className in imageLabels or not imageLabels[className] == classLabel:
                raise ValueError("Ground truth definitions do not match those in the class definitions")

        return numpy.sum(self.compute_nb_patches_per_image(-1, listgtPaths, classLabel))

    def create_patches(self, imagepath, patch_coords, className, imageName, gt=True, debug=False):

        svsImage = image_utils.open_svs_image_forced(imagepath)

        # debug propose
        if debug:
            imagesize = svsImage.get_level_dimension(self.lod)
            imageDebugPatches = numpy.zeros((imagesize[1], imagesize[0], 3), dtype=numpy.uint8)

        patch_shift = self.patch_size // 2
        for i, coords in enumerate(patch_coords):

            patch = svsImage.read_region((coords[0] - patch_shift, coords[1] - patch_shift), self.lod, (self.patch_size, self.patch_size))

            output = os.path.join(self.output_folder, "images", className, imageName + "_" + className + "_patch_" + str(i) + ".png")
            image_utils.save_image(patch, output)

            if debug:
                imageDebugPatches[coords[1] - patch_shift: coords[1] + patch_shift, coords[0] - patch_shift: coords[0] + patch_shift, 1] += 80

            sys.stdout.write('patch extracted : {:^5}\r'.format(i))

        print(" %d %s patches extracted" % (len(patch_coords), className))

        if debug:
            imageDebugPatches[:, :, 2] = 100 * gt[:, :]
            testImagePatchesPath = os.path.join(self.output_folder, 'image-patches', className, imageName + ".png")
            image_utils.save_image(imageDebugPatches, testImagePatchesPath)

    def create_folders(self, className, debug):
        """

        create_folders: create the differents folders used in the extraction of patches and the debugging


        :param className: (string) the name of the class
        :param debug: (bool) if true create the folder for the debug result
        """
        # Creation of the folder
        outputDir = self.output_folder

        # Create directory to write patches
        if not os.path.exists(os.path.join(outputDir, 'gts', className)):
            os.makedirs(os.path.join(outputDir, 'gts', className))
        if not os.path.exists(os.path.join(outputDir, 'images', className)):
            os.makedirs(os.path.join(outputDir, 'images', className))
        # Directory used for debugging purpose
        if debug and not os.path.exists(os.path.join(outputDir, 'image-patches', className)):
            os.makedirs(os.path.join(outputDir, 'image-patches', className))

    def normalise_image(self, image, tissuePath):
        """
        normalise_image normalise the image with the reinhard stain method, the normalization is where the tissue is
        nto on the negatives

        :param image: (numpy.array) the image to normalize
        :param tissuePath: (string) the path of the grouthtruth
        :return the image normalized
        """

        raise ValueError('Image normalisation has not been implemented when using disk based images')

        # if self.normalise_within_tissue:
        #
        #     tissue_mask = image_utils.read_binary_image(tissuePath)
        #
        #     normalised_image = image_utils.normalise_rgb_image_from_stat_file(image[tissue_mask > 0, :],
        #                                                                       self.h5py_norm_filename,
        #                                                                       self.normalise_within_tissue)
        #
        #     image[tissue_mask > 0] = normalised_image
        # else:
        #     image = image_utils.normalise_rgb_image_from_stat_file(image, self.h5py_norm_filename,
        #                                                            self.normalise_within_tissue)
        # return image

    def check_coordinate_is_valid(self, imageShape, patchSize, x, y):
        """
        check_coordinate_is_valid: checks whether a coordinate is too close to the border to extract a patch around it

        :param imageShape: (int, int) the size of the full image (y, x)
        :param patchSize: (int) the size of the patches to extract
        :param x: (int) the position X
        :param y: (int) the position Y
        :return: (boolean) True if the patch generated is not overflowing the image size
        """

        return (y >= int(patchSize / 2)) & \
               (y < imageShape[0] - int(patchSize / 2)) & \
               (x >= int(patchSize / 2)) & \
               (x < imageShape[1] - int(patchSize / 2))

    def label_valid_regions(self, gt, classLabel, patchSize):
        """
        label_valid_regions: returns labelled mask of connected components and a list of the labels from which patches can
        be extracted (based on their proximity to the image border)

        :param gt: numpy.array (int, int, int) the ground truth of the image (contains every class)
        :param classLabel: (int) the class label required
        :param patchSize: (int) the size of the patch
        :return: (list of skimage.regions, list of skimage.region.labels) the connected subregions of the classlabel in gt
        and the list of valid_label that can be extracted
        """

        gt[gt != classLabel] = 0
        #labelmask = skimage.measure.label(gt, background=0)  # nb of objects in this image

        s = scipy.ndimage.generate_binary_structure(2, 2)
        labelmask, _ = scipy.ndimage.label(gt, structure=s)

        valid_labels = []

        regions = skimage.measure.regionprops(labelmask)
        for region in regions:
            y, x = region.centroid
            if self.check_coordinate_is_valid(gt.shape, patchSize, int(x), int(y)):
                valid_labels.append(region.label)

        return regions, valid_labels

    def split_nb_patches_per_image(self, nbImages, nbObjectsPerImage, nbPatches):

        #  Construct the list of cumulative number of objects in each mask image.

        cumulativeNbObjectsPerImage = numpy.cumsum(nbObjectsPerImage)
        if cumulativeNbObjectsPerImage[-1] < nbPatches:
            raise ValueError("Not enough objects for the number of patches requested (found %s, requested %s)" % (cumulativeNbObjectsPerImage[-1], nbPatches))

        #  Random selection of "nbPatches" objects among objects in all masks
        labelSelected = numpy.random.choice(numpy.arange(1, cumulativeNbObjectsPerImage[-1] + 1), size=nbPatches, replace=False)

        #  Construct the list of the number of selected patches by mask image.
        nbPatchesPerImage = numpy.zeros(nbImages, numpy.int)
        lowBound = 0
        for i in range(nbImages):
            upBound = cumulativeNbObjectsPerImage[i]
            nbPatchesPerImage[i] = numpy.sum(numpy.logical_and(labelSelected > lowBound, labelSelected <= upBound))
            lowBound = upBound

        return nbPatchesPerImage
