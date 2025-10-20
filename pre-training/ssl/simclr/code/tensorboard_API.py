import io
import os

import cv2
import numpy as np

import tensorflow as tf
import albumentations as A

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def DeNormalize_wrt_Mean_Std(config, images):
    transform = A.Compose([A.Normalize(mean=0., std=1/config["data.stddev"], max_pixel_value=1.0,
                                       p=config["augmentations_prob.NormalizeMeanStd_p"]),
                           A.Normalize(mean=-config["data.mean"], std=1., max_pixel_value=1.0,
                                       p=config["augmentations_prob.NormalizeMeanStd_p"])])

    denormalize_images = np.zeros(images.shape[:])
    for batch in range(images.shape[0]):
        denormalize_images[batch, :, :, :] = transform(image=images[batch, :, :, :].numpy())["image"]
    return denormalize_images


class TensorboardLogs:
    def __init__(self, config):
        self.config = config
        self.logs_dir = os.path.join(self.config["output.OutputDir"], 'tensorboard', self.config["data.Mode"])
        self.train_writer = tf.summary.create_file_writer(self.logs_dir)

        self.Images_Dir = os.path.join(self.config["output.OutputDir"], "outputs", "Images")
        os.makedirs(self.Images_Dir, exist_ok=True)
        self.AugmentedImagesView1_Dir = os.path.join(self.config["output.OutputDir"], "outputs", "AugmentedImagesView1")
        os.makedirs(self.AugmentedImagesView1_Dir, exist_ok=True)
        self.AugmentedImagesView2_Dir = os.path.join(self.config["output.OutputDir"], "outputs", "AugmentedImagesView2")
        os.makedirs(self.AugmentedImagesView2_Dir, exist_ok=True)

    # define a function which returns an image as numpy array from figure
    def figure2image(self, fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.reshape(img, (-1, img.shape[0], img.shape[1], img.shape[2]))

    def images_plot(self, images, augmented=False, mode="save", name=""):
        if augmented:
            images = DeNormalize_wrt_Mean_Std(config=self.config, images=images)

        # if mode == "save":
        #     print(f"name: {name} --- {np.amin(images)} --- {np.amax(images)}")

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

        if mode == "save":
            if name.split("_")[0] == "Images":
                figure.savefig(os.path.join(self.Images_Dir, f"{name}.png"))
            elif name.split("_")[0] == "AugmentedImagesView1":
                figure.savefig(os.path.join(self.AugmentedImagesView1_Dir, f"{name}.png"))
            elif name.split("_")[0] == "AugmentedImagesView2":
                figure.savefig(os.path.join(self.AugmentedImagesView2_Dir, f"{name}.png"))
            else:
                raise ValueError("Directories are not properly configured.")

            plt.close(figure)

        elif mode == "tensorboard":
            fig2img = self.figure2image(figure)
            plt.close(figure)
            return fig2img
        else:
            raise ValueError("Please specify one of the modes: {save | tensorboard}")

    # Save to directory
    def save_to_dir(self, dataset, epoch):
        for images, aug_view1, aug_view2 in dataset.take(1):
            # save images and augmented views of images to saveDir
            self.images_plot(images=images, augmented=False, mode="save", name=f"Images_at_Epoch_{epoch}")
            self.images_plot(images=aug_view1, augmented=True, mode="save", name=f"AugmentedImagesView1_at_Epoch_{epoch}")
            self.images_plot(images=aug_view2, augmented=True, mode="save", name=f"AugmentedImagesView2_at_Epoch_{epoch}")
        del dataset, images, aug_view1, aug_view2

    # Save to Tensorboard
    def save_to_tensorboard(self, dataset, epoch):
        for images, aug_view1, aug_view2 in dataset.take(1):
            # save images and augmented views of images to Tensorboard Logs
            images = self.images_plot(images=images, augmented=False, mode="tensorboard",
                                      name=f"Images_at_Epoch_{epoch}")
            aug_view1 = self.images_plot(images=aug_view1, augmented=True, mode="tensorboard",
                                         name=f"AugmentedImagesView1_at_Epoch_{epoch}")
            aug_view2 = self.images_plot(images=aug_view2, augmented=True, mode="tensorboard",
                                         name=f"AugmentedImagesView1_at_Epoch_{epoch}")
            with self.train_writer.as_default():
                tf.summary.image("Images", images, step=epoch)
                tf.summary.image("AugmentedImagesView1", aug_view1, step=epoch)
                tf.summary.image("AugmentedImagesView2", aug_view2, step=epoch)
        del dataset, images, aug_view1, aug_view2
