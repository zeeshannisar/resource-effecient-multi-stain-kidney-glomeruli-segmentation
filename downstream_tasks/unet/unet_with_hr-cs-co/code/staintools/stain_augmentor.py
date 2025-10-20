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

    def fit(self, I, staincode=None):
        """
        Fit to an image I.

        :param I:
        :return:
        """
        self.image_shape = I.shape
        if self.extractor.__name__ == 'RuifrokJohnstonStainExtractor':
            self.stain_matrix, self.zeroedchannels = self.extractor.get_stain_matrix(I, staincode)
        else:
            self.stain_matrix = self.extractor.get_stain_matrix(I)
            self.zeroedchannels = []
        self.source_concentrations = self.extractor.get_concentrations(I, self.stain_matrix)
        self.n_stains = self.source_concentrations.shape[1] - len(self.zeroedchannels)
        self.tissue_mask = get_luminosity_mask(I).ravel()

    def transform(self):
        """
        Transform to produce an augmented version of the fitted image.

        :return:
        """
        augmented_concentrations = copy.deepcopy(self.source_concentrations)

        for i in range(self.n_stains):

            if i in self.zeroedchannels:
                continue

            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(-self.sigma2, self.sigma2)
            if self.include_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= alpha
                augmented_concentrations[self.tissue_mask, i] += beta

        if self.extractor.__name__ == 'RuifrokJohnstonStainExtractor':
            I_augmented = np.exp(-(np.dot(augmented_concentrations, self.stain_matrix) - 255) * np.log(255) / 255) - 1
            I_augmented = np.clip(I_augmented, 0, 255).reshape(self.image_shape)
        else:
            I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, self.stain_matrix))
            I_augmented = I_augmented.reshape(self.image_shape)
            I_augmented = np.clip(I_augmented, 0, 255)

        return I_augmented

    def separate(self, I, staincode=None):
        image_shape = I.shape
        if self.extractor.__name__ == 'RuifrokJohnstonStainExtractor':
            stain_matrix, zeroedchannels = self.extractor.get_stain_matrix(I, staincode)
        else:
            stain_matrix = self.extractor.get_stain_matrix(I)
            zeroedchannels = []
        concentrations = self.extractor.get_concentrations(I, stain_matrix).reshape(image_shape)
        if staincode != "03":
            H_channel, O_channel = concentrations[:, :, 0], concentrations[:, :, 1]
        else:
            H_channel, O_channel = concentrations[:, :, 0], concentrations[:, :, 1] + concentrations[:, :, 2]

        return np.expand_dims(H_channel, axis=-1), np.expand_dims(O_channel, axis=-1)

