import os

import numpy as np

import albumentations as A
import tensorflow.keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def Standardise(image):
    return (image - image.min()) / ((image.max() - image.min()) + K.epsilon())


def RescalePixelValues(config, images):
    denormalize_images = np.zeros(images.shape[:])
    for batch in range(images.shape[0]):
        denormalize_images[batch, :, :, :] = Standardise(image=images[batch, :, :, :].numpy())
    return denormalize_images


def images_plot(config, images, augmented=False, name=None):
    data_samples_dir = os.path.join(config["output.OutputDir"], "view_data_samples", )
    os.makedirs(data_samples_dir, exist_ok=True)
    if augmented:
        images = RescalePixelValues(config=config, images=images)

    # print(f"name: {name} --- {np.amin(images)} --- {np.amax(images)}")

    # x should be in (BatchSize, H, W, C)
    assert images.ndim == 4

    size = int(np.ceil(np.sqrt(images.shape[0])))
    figure = plt.figure(figsize=(size*2, size*2))
    figure.suptitle(f"{name.replace('_', ' ')}", fontsize=12)
    for i in range(images.shape[0]):
        plt.subplot(size, size, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # if grayscale
        if images.shape[-1] == 1:
            plt.imshow(images[i], cmap=plt.cm.binary)
        else:
            plt.imshow(images[i])

    figure.savefig(os.path.join(data_samples_dir, f"{name}.png"))
    plt.close(figure)