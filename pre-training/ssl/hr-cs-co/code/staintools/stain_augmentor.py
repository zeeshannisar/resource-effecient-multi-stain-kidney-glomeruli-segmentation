from __future__ import division

import numpy as np
import copy

from staintools.stain_extractors.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extractors.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.stain_extractors.ruifrokjohnston_stain_extractor import RuifrokJohnstonStainExtractor
from staintools.utils.misc_utils import get_luminosity_mask


class StainAugmentor(object):

    def __init__(self, method, sigma1=0.2, sigma2=0.2, include_background=True):
        if method.lower() == 'macenko':
            self.extractor = MacenkoStainExtractor
        elif method.lower() == 'vahadane':
            self.extractor = VahadaneStainExtractor
        elif method.lower() == 'ruifrokjohnston':
            self.extractor = RuifrokJohnstonStainExtractor
        else:
            raise Exception('Method not recognized.')
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.include_background = include_background

    def separate(self, image, stain_code=None):
        if self.extractor.__name__ == 'RuifrokJohnstonStainExtractor':
            stain_matrix, zeroed_channels = self.extractor.get_stain_matrix(image, stain_code)
        else:
            stain_matrix = self.extractor.get_stain_matrix(image)
            zeroed_channels = []

        source_concentrations = self.extractor.get_concentrations(image, stain_matrix)
        return source_concentrations, stain_matrix, zeroed_channels

    def perturb(self, image, source_concentrations, stain_matrix, zeroed_channels):
        n_stains = source_concentrations.shape[1] - len(zeroed_channels)
        tissue_mask = get_luminosity_mask(image).ravel()
        augmented_concentrations = copy.deepcopy(source_concentrations)

        for i in range(n_stains):
            if i in zeroed_channels:
                continue

            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(-self.sigma2, self.sigma2)
            if self.include_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[tissue_mask, i] *= alpha
                augmented_concentrations[tissue_mask, i] += beta

        image_augmented = np.exp(-(np.dot(augmented_concentrations, stain_matrix) - 255) * np.log(255) / 255) - 1
        image_augmented = np.clip(image_augmented, 0, 255)
        return augmented_concentrations, image_augmented
