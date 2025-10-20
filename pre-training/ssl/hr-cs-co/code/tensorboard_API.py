import io
import os

import cv2
import numpy as np

import tensorflow as tf
import albumentations as A
import tensorflow.keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def DeNormalize_wrt_Mean_Std(config, images):
    transform = A.Compose([A.Normalize(mean=0., std=1/config["data.stddev"], max_pixel_value=1.0, p=1.0),
                           A.Normalize(mean=-config["data.mean"], std=1., max_pixel_value=1.0, p=1.0)])

    denormalize_images = np.zeros(images.shape[:])
    for batch in range(images.shape[0]):
        denormalize_images[batch, :, :, :] = transform(image=images[batch, :, :, :].numpy())["image"]
    return denormalize_images


def Standardise(image):
    return (image - image.min()) / ((image.max() - image.min()) + K.epsilon())


def RescalePixelValues(images):
    denormalize_images = np.zeros(images.shape[:])
    for batch in range(images.shape[0]):
        denormalize_images[batch, :, :, :] = Standardise(image=images[batch, :, :, :].numpy())
    return denormalize_images


def ExtractCentralRegion(images, shape=(324, 324)):
    x = shape[0]
    y = shape[1]

    images_x_offset = (images.shape[0] - x) // 2
    images_y_offset = (images.shape[1] - y) // 2
    images = images[images_x_offset:images.shape[0] - images_x_offset, images_y_offset:images.shape[1] - images_y_offset]
    return images


def resize(images, size=None):
    return tf.image.resize(images=images, size=size, method=tf.image.ResizeMethod.BILINEAR,
                           preserve_aspect_ratio=False, antialias=False, name=None)


class TensorboardLogs:
    def __init__(self, config):
        self.config = config
        self.logs_dir = os.path.join(self.config["output.OutputDir"], 'tensorboard', 'logs')
        if self.config["training.SSLModelPhase"] == "cross_stain_prediction":
            self.train_loss_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, "train_loss"))
            self.valid_loss_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, "valid_loss"))
            self.outputs_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, "visual_outputs"))
        if self.config["training.SSLModelPhase"] == "contrastive_learning":
            self.train_loss_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, "train_loss"))
            self.valid_loss_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, "valid_loss"))
            self.outputs_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, "visual_outputs"))

        self.Outputs_Dir = os.path.join(self.config["output.OutputDir"], "visual_outputs")
        os.makedirs(self.Outputs_Dir, exist_ok=True)

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

    def images_plot(self, images_H, H2O_predictions, labels_O, images_O, O2H_predictions, labels_H, mode="save", name=""):
        rows = 5
        cols = 6
        figure = plt.figure(figsize=(18, 16))
        figure.subplots_adjust(wspace=0, hspace=0)
        font_size = 20
        cnt = 1
        for r in range(rows):
            for c in range(cols):
                ax = figure.add_subplot(rows, cols, cnt)
                if c == 0:
                    ax.imshow(images_H[r, :, :, 0], cmap="gray")
                    ax.set_title("image_H", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                    ax.axis("off")
                if c == 1:
                    ax.imshow(H2O_predictions[r, :, :, 0], cmap="gray")
                    ax.set_title("predicted_H2O", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                    ax.axis("off")
                if c == 2:
                    ax.imshow(labels_O[r, :, :, 0], cmap="gray")
                    ax.set_title("ground-truth_O", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                    ax.axis("off")
                if c == 3:
                    ax.imshow(images_O[r, :, :, 0], cmap="gray")
                    ax.set_title("image_O", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                    ax.axis("off")
                if c == 4:
                    ax.imshow(O2H_predictions[r, :, :, 0], cmap="gray")
                    ax.set_title("predicted_O2H", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                    ax.axis("off")
                if c == 5:
                    ax.imshow(labels_H[r, :, :, 0], cmap="gray")
                    ax.set_title("ground-truth_H", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                    ax.axis("off")
                cnt += 1

        figure.tight_layout()
        if mode == "save":
            figure.savefig(os.path.join(self.Outputs_Dir, name+".png"), bbox_inches="tight")
            plt.close(figure)

        elif mode == "tensorboard":
            fig2img = self.figure2image(figure)
            plt.close(figure)
            return fig2img
        else:
            raise ValueError("Please specify one of the modes: {save | tensorboard}")

    # Saving Visual Outputs to specified directory
    def save_to_dir(self, images_H, images_O, labels_H, labels_O, H2O_predictions, O2H_predictions, name):
        images_H = RescalePixelValues(images=images_H)
        images_O = RescalePixelValues(images=images_O)
        images_H = np.array([ExtractCentralRegion(x, shape=H2O_predictions.shape[1:-1]) for x in images_H])
        labels_O = np.array([ExtractCentralRegion(x, shape=H2O_predictions.shape[1:-1]) for x in labels_O])

        images_O = np.array([ExtractCentralRegion(x, shape=O2H_predictions.shape[1:-1]) for x in images_O])
        labels_H = np.array([ExtractCentralRegion(x, shape=O2H_predictions.shape[1:-1]) for x in labels_H])

        self.images_plot(images_H, H2O_predictions, labels_O, images_O, O2H_predictions, labels_H,
                         mode="save", name=name)

    # Saving Visual Outputs to TensorBoard Logs
    def save_to_tensorboard(self, images_H, images_O, labels_H, labels_O, H2O_predictions, O2H_predictions, epoch):
        images_H = RescalePixelValues(images=images_H)
        images_O = RescalePixelValues(images=images_O)
        images_H = np.array([ExtractCentralRegion(x, shape=H2O_predictions.shape[1:-1]) for x in images_H])
        labels_O = np.array([ExtractCentralRegion(x, shape=H2O_predictions.shape[1:-1]) for x in labels_O])

        images_O = np.array([ExtractCentralRegion(x, shape=O2H_predictions.shape[1:-1]) for x in images_O])
        labels_H = np.array([ExtractCentralRegion(x, shape=O2H_predictions.shape[1:-1]) for x in labels_H])

        # save images and augmented views of images to Tensorboard Logs
        image = self.images_plot(images_H, H2O_predictions, labels_O, images_O, O2H_predictions, labels_H,
                                 mode="tensorboard", name=f"epoch_{epoch}")
        with self.outputs_writer.as_default():
            tf.summary.image("visual_outputs", image, step=epoch)
