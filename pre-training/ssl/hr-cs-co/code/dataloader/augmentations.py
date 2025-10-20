import numpy as np
import albumentations as A


def which_custom_augmentations(config):
    transforms = []

    # Flip the input either horizontally, vertically or both horizontally and vertically.
    if config["augmentations.Flip"]:
        transforms.append(A.Flip(p=config["augmentations_prob.Flip_p"]))

    # Torchvision variant of crop a random part of the input and rescale it to some size.
    if config["augmentations.RandomResizedCrop"]:
        transforms.append(A.RandomResizedCrop(height=config["training.CroppedImageSize"][0],
                                              width=config["training.CroppedImageSize"][1],
                                              scale=config["augmentations.Scale"],
                                              p=config["augmentations_prob.RandomResizedCrop_p"]))

    # Blur the input image using a Gaussian filter with a random kernel size.
    if config["augmentations.GaussianBlur"]:
        transforms.append(A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(1.0, 2.0),
                                         p=config["augmentations_prob.GaussianBlur_p"]))

    return transforms

