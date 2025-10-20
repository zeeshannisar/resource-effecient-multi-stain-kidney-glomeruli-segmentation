from utils import image_utils
import numpy
import warnings

from patch_extractors.extractor import ExtractMethod


class RandomExtractMethod(ExtractMethod):

    DEFAULTMAX = 2500

    def __init__(self, file_path, output_folder, lod, patch_size, h5py_norm_filename, normalise_within_tissue):
        super(RandomExtractMethod, self).__init__(file_path, output_folder, lod, patch_size, h5py_norm_filename,
                                               normalise_within_tissue)
        self.default_max = RandomExtractMethod.DEFAULTMAX

    def extract_patches(self, patients, className, classLabel, nbPatches, debug=False):

        print(" -------------- Create Random Patches -------------- ")
        return super(RandomExtractMethod, self).extract_patches(patients, className, classLabel, nbPatches, debug)

    def compute_nb_patches_per_image(self, nbPatch, listgtpaths, classLabel):
        """

        compute_nb_random_patches_per_image: calculates the number of patches to extract for each image, if the number
        of patches can not be divided by the number of images the remaining patches are randomly distributed between
        the images

        :param nbPatches: (int) the number of patches required
        :param nbImages: (int) the number of images
        :return: (list of int) the number of patches to extract for each image
        """

        if nbPatch == -1:
            nbPatch = self.default_max

        nbImages = len(listgtpaths)

        x = int(nbPatch / nbImages)
        nbPatchesPerImage = [x] * nbImages

        nbPatches_remaining = nbPatch - (x * nbImages)
        c = numpy.random.choice(nbImages, nbPatches_remaining, replace=False)
        for i in c:
            nbPatchesPerImage[i] += 1

        return nbPatchesPerImage

    def get_patch_coords(self, gt, classLabel, nb):

        # Compute class coordinates
        coordY, coordX = numpy.nonzero(gt == classLabel)

        oob_ind = self.check_coordinate_is_valid(gt.shape, self.patch_size, coordX, coordY)
        coordY = coordY[oob_ind]
        coordX = coordX[oob_ind]

        #  Select the first "nbPatchPerImage" class pixels and create a patch centered on each one.
        small_patch_shift = self.patch_size // 3
        ind_unused = numpy.ones(coordY.size, dtype=numpy.bool)
        patchcoords = []
        while (nb == -1 or len(patchcoords) < nb) and numpy.any(ind_unused):

            valid_inds = numpy.where(ind_unused)[0]
            if valid_inds.size > 0:
                ind = numpy.random.randint(valid_inds.size)
                y = int(coordY[valid_inds[ind]])
                x = int(coordX[valid_inds[ind]])
                small_patch = gt[y - small_patch_shift: y + small_patch_shift, x - small_patch_shift: x + small_patch_shift]
                ind_unused[ind] = False

                while not numpy.all(small_patch == classLabel) and numpy.any(ind_unused):
                    valid_inds = numpy.where(ind_unused)[0]
                    if valid_inds.size > 0:
                        ind = numpy.random.randint(valid_inds.size)
                        y = int(coordY[valid_inds[ind]])
                        x = int(coordX[valid_inds[ind]])
                        small_patch = gt[y - small_patch_shift: y + small_patch_shift, x - small_patch_shift: x + small_patch_shift]
                        ind_unused[ind] = False

                if numpy.all(small_patch == classLabel):
                    patchcoords.append((x, y))

        return patchcoords
