import cv2
import math
import numpy as np
import albumentations as A


def rgb2gray_function(image, **kwargs):
    return np.stack((np.dot(image[:, :, :], [0.2989, 0.5870, 0.1140]),) * 3, axis=-1)


def solarize_function(image, **kwargs):
    return np.where(image < 0.5, image, 1. - image)


def which_custom_augmentations(config):
    transforms = []

    # Random Cropping: a random patch of the image is selected, with an area uniformly sampled between 10% and 100%
    # of that of the original image, and an aspect ratio logarithmically sampled between 3 / 4 and 4 / 3.
    # This patch is then resized to the target size of 224 × 224 using bicubic interpolation.
    if config["augmentations.RandomResizedCrop"]:
        transforms.append(A.RandomResizedCrop(height=config["training.CroppedImageSize"][0],
                                              width=config["training.CroppedImageSize"][1],
                                              scale=(0.1, 1.0), ratio=(0.75, 1.3333333333333333),
                                              interpolation=cv2.INTER_CUBIC,
                                              p=config["augmentations_prob.RandomResizedCrop_p"]))

    # Flip the input either horizontally, vertically or both horizontally and vertically.
    if config["augmentations.Flip"]:
        transforms.append(A.Flip(p=config["augmentations_prob.Flip_p"]))

    # Randomly changes the brightness, contrast, and saturation of an image.
    # Compared to ColorJitter from torchvision, this transform gives a little bit different results because
    # Pillow (used in torchvision) and OpenCV (used in Albumentations) transform an image to HSV format by
    # different formulas. Another difference - Pillow uses uint8 overflow, but we use value saturation.
    if config["augmentations.ColorJitter"]:
        transforms.append(A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
                                        p=config["augmentations_prob.ColorJitter_p"]))

    # Convert the image from RGB to GrayScale.
    if config["augmentations.RGB2GRAY"]:
        rgb2gray_transform = A.Lambda(name='RGB2GRAY', image=rgb2gray_function,
                                      p=config["augmentations_prob.RGB2GRAY_p"])
        transforms.append(rgb2gray_transform)

    # Blur the input image using a Gaussian filter with a random kernel size.
    if config["augmentations.GaussianBlur"]:
        kernel_size = math.ceil(config["training.CroppedImageSize"][0] / config["augmentations.GaussianBlurDivider"])
        transforms.append(A.GaussianBlur(blur_limit=kernel_size, sigma_limit=(0.1, 2),
                                         p=config["augmentations_prob.GaussianBlur_p"]))

    # Invert the image if > than a threshold.
    if config["augmentations.Solarize"]:
        rgb2gray_transform = A.Lambda(name='Solarize', image=solarize_function,
                                      p=config["augmentations_prob.Solarize_p"])
        transforms.append(rgb2gray_transform)

    if config["augmentations.GridDistort"]:
        transforms.append(A.GridDistortion(num_steps=9, distort_limit=0.3, interpolation=1,
                                           border_mode=2, p=config["augmentations_prob.GridDistort_p"]))

    if config["augmentations.GridShuffle"]:
        transforms.append(A.RandomGridShuffle(grid=(3, 3), always_apply=False,
                                              p=config["augmentations_prob.GridShuffle_p"]))

    if config["preprocessing.Normalise"]:
        transforms.append(A.Normalize(mean=config["data.mean"], std=config["data.stddev"],
                                      always_apply=True, max_pixel_value=1.0, p=1.0))

    return transforms

