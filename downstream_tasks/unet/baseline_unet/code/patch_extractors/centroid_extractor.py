from utils import image_utils
import numpy
import warnings

from patch_extractors.extractor import ExtractMethod


class CentroidExtractor(ExtractMethod):

    def __init__(self, file_path, output_folder, lod, patch_size, h5py_norm_filename, normalise_within_tissue):
        super(CentroidExtractor, self).__init__(file_path, output_folder, lod, patch_size, h5py_norm_filename, normalise_within_tissue)

    def extract_patches(self, patients, className, classLabel, nbPatches, debug=False):
        print(" -------------- Create Centroid Patches -------------- ")
        return super(CentroidExtractor, self).extract_patches(patients, className, classLabel, nbPatches, debug)

    def compute_nb_patches_per_image(self, nbPatch, listgtpaths, classLabel):
        """
        compute_nb_centred_patches_per_image: compute for each mask (of the same class), the number of patches to extract
        in order to obtain the "nbPatch" for the specified class.

        :param gts: (list of (int, string)) the list of ground truths with their paths

        :param patchSize: (int) the patch size
        :param nbPatches: (int) the total number of patches required
        :param classLabel: (int) the label of the class which the mask belongs to
        :return: an array containing the number of patches to extract for each mask
        """

        #  Construct the list of cumulative number of objects in each mask image.
        nbImages = len(listgtpaths)
        nbObjectsPerImage = numpy.zeros(nbImages, numpy.int)
        for i, gtpath in enumerate(listgtpaths):
            gt = image_utils.read_image(gtpath).astype(numpy.uint8)
            _, labels = self.label_valid_regions(gt, classLabel, self.patch_size)
            nbObjectsPerImage[i] = len(labels)

        if nbPatch == -1:
            nbPatchesPerImage = nbObjectsPerImage
        else:
            nbPatchesPerImage = self.split_nb_patches_per_image(nbImages, nbObjectsPerImage, nbPatch)

        return nbPatchesPerImage

    def get_patch_coords(self, gt, classLabel, nb):
        """

                extract_patches_from_image_centroid: extract a class's patches from the image and the masks. For every connected
                region of the mask, extract one patch at its center.

                :param classlabel: (int) the label of the class to extract
                :param gt: (numpy.array of shape Y * X (int)) the image's ground truth
                :param nb: (int) the number of patches
                """

        # print("Label %d %s" % (classLabel, className))
        regions, labels = self.label_valid_regions(gt, classLabel, self.patch_size)

        #  Select randomly "nbPatch" structures among all the structures in the GT
        if nb != -1:
            if nb <= len(labels):
                selectedLabelsIdx = numpy.random.permutation(len(labels))[:nb]
            else:
                raise ValueError("Not enough samples in class %s for the number of patches required" % classLabel)
        else:
            selectedLabelsIdx = numpy.random.permutation(len(labels))
        selectedLabels = set([labels[i] for i in selectedLabelsIdx])

        regions = numpy.random.permutation(regions)
        patchcoords = []
        for region in regions:
            if region.label in selectedLabels:
                y, x = region.centroid
                patchcoords.append((int(x), int(y)))

        return patchcoords
