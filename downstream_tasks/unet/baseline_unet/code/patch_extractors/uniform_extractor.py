"""
uniform_extractor.py : Implementation of the ExtractMethod for the uniform extraction
"""
from utils import image_utils
import numpy
import warnings

from patch_extractors.extractor import ExtractMethod

VALIDPATCHFRACTION = 0.15

class UniformExtractor(ExtractMethod):
    """
    UniformExtractor : implementation of the ExtractMethod
     Extraction of the patches contained in the differents connected elements
        -The patches are uniformly taken from the groundtruth, they form a grid
        -the patches overlap with a margin called stride
        -patches that do not contains enough groundtruth of the asked label are ignored
        -if a connected element has no created patches, a default centroid patch is created for it
        -The maximum number of patches depends on the number of connected element in the groundtruth the size of the patches
        and the value of the stride

        -the number of patches requested  is inferior to the number possible, the extraction choose them randomly
    """

    def __init__(self, file_path, output_folder, lod, patch_size, h5py_norm_filename, normalise_within_tissue, overlap=10):
        super(UniformExtractor, self).__init__(file_path, output_folder, lod, patch_size, h5py_norm_filename, normalise_within_tissue)
        self.stride = self.patch_size - overlap

    def extract_patches(self, patients, className, classLabel, nbPatches, debug=False):
        print(" -------------- Create Uniform Patches -------------- ")
        # import ipdb; ipdb.set_trace()
        return super(UniformExtractor, self).extract_patches(patients, className, classLabel, nbPatches, debug)

    def compute_nb_patches_per_image(self, nbPatch, listgtpaths, classLabel):
        """
           compute_nb_random_patches_per_image: calculates the number of patches to extract for each image, if the number
           of patches can not be divided by the number of images the remaining patches are randomly distributed between
           the images

           :param nbPatches: (int) the number of patches required
           :param nbImages: (int) the number of images
           :return: (list of int) the number of patches to extract for each image
           """
        #import ipdb; ipdb.set_trace()

        nbImages = len(listgtpaths)
        patchMaxPerImage = numpy.zeros(nbImages, numpy.int)
        for i, gtpath in enumerate(listgtpaths):
            gt = image_utils.read_image(gtpath).astype(numpy.uint8)
            patchMaxPerImage[i] = len(self.get_patch_coords(gt, classLabel, -1))

        # Only if the number of patches is undefined we calculated the number of all possibles patches
        if nbPatch == -1:
            nbPatchesPerImage = patchMaxPerImage
        else:
            # if a number is given
            nbPatchesPerImage = self.split_nb_patches_per_image(nbImages, patchMaxPerImage, nbPatch)

        return nbPatchesPerImage

    def get_patch_coords(self, gt, classLabel, nb):
        """
        get_patch_coords : get the number of patches uniform that can be calculated on the groundtruth for one class
        On a first step the function extracts the different connected regions of the class
        The patches are based on the bbox of the region, the fist one begin on the left top corner.
        The others patches are generated in the right and bottom directions.

        Every regions have at least 1 patch, but some patches can be ignored if the groundtruth put inside is inexistant or
        too low.
        For the small region the patches position is centred in it centroid instead of the bottom top corner


        :param gt: (numpy.array of int) the groundtruth image
        :param classLabel: (int) the class number to extract from the groundtruth
        :param patchSize: (int) the size of the patches used
        :return:  list of (int,int) the list of coordinates (based on this center) of the valid patches
        """

        regions, labels = self.label_valid_regions(gt, classLabel, self.patch_size)

        patch_shift = self.patch_size // 2

        patchcoords = []
        for region in regions:
            patchExtracted = False
            minX, minY, maxX, maxY = region.bbox
            for y_pos in numpy.arange(minX+patch_shift, maxX-patch_shift, self.stride):
                for x_pos in numpy.arange(minY+patch_shift, maxY-patch_shift, self.stride):
                    if self.check_coordinate_is_valid(gt.shape, self.patch_size, x_pos, y_pos) and \
                            numpy.mean(gt[y_pos - patch_shift:y_pos + patch_shift, x_pos - patch_shift: x_pos + patch_shift]) >= UniformExtractor.VALIDPATCHFRACTION:
                        patchcoords.append((x_pos, y_pos))
                        patchExtracted = True

            if not patchExtracted:
                y, x = region.centroid
                patchcoords.append((int(x), int(y)))

        if nb != -1:
            if len(patchcoords) >= nb:
                patchcoords = numpy.random.permutation(patchcoords)[:nb]
            else:
                raise ValueError("Not enough samples in class %s for the number of patches required" % classLabel)

        return patchcoords
