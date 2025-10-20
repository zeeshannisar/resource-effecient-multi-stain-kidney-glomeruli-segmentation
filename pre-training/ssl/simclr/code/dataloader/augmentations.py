import cv2
import math
import numpy as np
import albumentations as A


def rgb2gray_function(image, **kwargs):
    return np.stack((np.dot(image[:, :, :], [0.2989, 0.5870, 0.1140]),) * 3, axis=-1)


def which_custom_augmentations(config):
    transforms = []

    # Torchvision variant of crop a random part of the input and rescale it to some size.
    if config["augmentations.RandomResizedCrop"] and not config["training.Evaluate"]:
        transforms.append(A.RandomResizedCrop(height=config["training.CroppedImageSize"][0],
                                              width=config["training.CroppedImageSize"][1],
                                              scale=config["augmentations.Scale"],
                                              p=config["augmentations_prob.RandomResizedCrop_p"]))

    # Flip the input either horizontally, vertically or both horizontally and vertically.
    if config["augmentations.Flip"] and not config["training.Evaluate"]:
        transforms.append(A.Flip(p=config["augmentations_prob.Flip_p"]))

    # Randomly changes the brightness, contrast, and saturation of an image.
    # Compared to ColorJitter from torchvision, this transform gives a little bit different results because
    # Pillow (used in torchvision) and OpenCV (used in Albumentations) transform an image to HSV format by
    # different formulas. Another difference - Pillow uses uint8 overflow, but we use value saturation.
    if config["augmentations.ColorJitter"] and not config["training.Evaluate"]:
        transforms.append(A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8,
                                        hue=0.2, p=config["augmentations_prob.ColorJitter_p"]))

    # Convert the image from RGB to GrayScale.
    if config["augmentations.RGB2GRAY"] and not config["training.Evaluate"]:
        rgb2gray_transform = A.Lambda(name='RGB2GRAY', image=rgb2gray_function,
                                      p=config["augmentations_prob.RGB2GRAY_p"])
        transforms.append(rgb2gray_transform)



    # Blur the input image using a Gaussian filter with a random kernel size.
    if config["augmentations.GaussianBlur"] and not config["training.Evaluate"]:
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2),
                                         p=config["augmentations_prob.GaussianBlur_p"]))
    # Randomly rotate the input by 90 degrees zero or more times.
    if config["augmentations.RandomRotate90"] and not config["training.Evaluate"]:
        transforms.append(A.RandomRotate90(p=config["augmentations_prob.RandomRotate90_p"]))

    if config["augmentations.GridDistort"] and not config["training.Evaluate"]:
        transforms.append(A.GridDistortion(num_steps=9, distort_limit=0.3, interpolation=1,
                                           border_mode=2, p=config["augmentations_prob.GridDistort_p"]))

    if config["augmentations.BrightnessContrast"] and not config["training.Evaluate"]:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=config["augmentations.brightness"],
                                                     contrast_limit=config["augmentations.contrast"],
                                                     p=config["augmentations_prob.BrightnessContrast_p"]))

    if config["augmentations.GridShuffle"] and not config["training.Evaluate"]:
        transforms.append(A.RandomGridShuffle(grid=(3, 3), always_apply=False,
                                              p=config["augmentations_prob.GridShuffle_p"]))

    if config["augmentations.NormalizeMeanStd"]:
        transforms.append(A.Normalize(mean=config["data.mean"], std=config["data.stddev"], always_apply=True,
                                      max_pixel_value=1.0, p=config["augmentations_prob.NormalizeMeanStd_p"]))

    return transforms

