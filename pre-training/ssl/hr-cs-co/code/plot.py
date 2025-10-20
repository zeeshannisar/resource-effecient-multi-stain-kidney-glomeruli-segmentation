import os
import numpy as np
import matplotlib.pyplot as plt


def ExtractLabelCentralRegion(images, required_shape):
    x = required_shape[0]
    y = required_shape[1]

    images_x_offset = (images.shape[0] - x) // 2
    images_y_offset = (images.shape[1] - y) // 2
    images = images[images_x_offset:images.shape[0] - images_x_offset,
             images_y_offset:images.shape[1] - images_y_offset]
    return images


def stain_separation_plot(x, H, O, savePath, stain_code=None):
    num_rows = 5
    num_columns = 3
    fig = plt.figure(figsize=(15, 25))
    fig.subplots_adjust(wspace=0, hspace=0)
    fontsize = 25
    cnt = 1
    for r in range(num_rows):
        for c in range(num_columns):
            ax = fig.add_subplot(num_rows, num_columns, cnt)
            if c == 0:
                ax.imshow(x[r, :, :, :])
                ax.set_title("Input", fontdict={'fontsize': fontsize, 'fontweight': 'medium'})
                ax.axis("off")

            if c == 1:
                ax.imshow(H[r, :, :], cmap="gray")
                ax.set_title("Haematoxyln", fontdict={'fontsize': fontsize, 'fontweight': 'medium'})
                ax.axis("off")

            if c == 2:
                ax.imshow(O[r, :, :], cmap="gray")
                ax.set_title("Others", fontdict={'fontsize': fontsize, 'fontweight': 'medium'})
                ax.axis("off")

            cnt += 1

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(savePath, f"stain_separation_{stain_code}_samples.png"), bbox_inches="tight")


def images_with_labels_plot(images_H, labels_O, images_O, labels_H, output_dir=None, stain_code=None):
    rows = 5
    cols = 4
    figure = plt.figure(figsize=(10, 14))
    figure.subplots_adjust(wspace=0, hspace=0)
    font_size = 15
    cnt = 1
    for r in range(rows):
        for c in range(cols):
            ax = figure.add_subplot(rows, cols, cnt)
            if c == 0:
                ax.imshow(images_H[r, :, :, 0], cmap="gray")
                ax.set_title("image_H", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 1:
                ax.imshow(labels_O[r, :, :, 0], cmap="gray")
                ax.set_title("ground-truth_O", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 2:
                ax.imshow(images_O[r, :, :, 0], cmap="gray")
                ax.set_title("image_O", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 3:
                ax.imshow(labels_H[r, :, :, 0], cmap="gray")
                ax.set_title("ground-truth_H", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            cnt += 1

    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"data_w_labels_{stain_code}_samples.png"), bbox_inches="tight")
    plt.close(figure)


def images_with_labels_plus_augmentations_plot(images, image_perturbed, images_H_T1, labels_O_T1, images_O_T1,
                                               labels_H_T1, images_H_T2, labels_O_T2, images_O_T2, labels_H_T2,
                                               output_dir=None, stain_code=None):
    rows = 5
    cols = 10
    figure = plt.figure(figsize=(30, 16))
    figure.subplots_adjust(wspace=0, hspace=0)
    font_size = 15
    cnt = 1
    for r in range(rows):
        for c in range(cols):
            ax = figure.add_subplot(rows, cols, cnt)
            if c == 0:
                ax.imshow(images[r])
                ax.set_title("input", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 1:
                ax.imshow(image_perturbed[r])
                ax.set_title("input_perturbed", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 2:
                ax.imshow(images_H_T1[r, :, :, 0], cmap="gray")
                ax.set_title("image_H_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 3:
                ax.imshow(labels_O_T1[r, :, :, 0], cmap="gray")
                ax.set_title("ground-truth_O_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 4:
                ax.imshow(images_H_T2[r, :, :, 0], cmap="gray")
                ax.set_title("image_H_T2", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 5:
                ax.imshow(labels_O_T2[r, :, :, 0], cmap="gray")
                ax.set_title("ground-truth_O_T2", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 6:
                ax.imshow(images_O_T1[r, :, :, 0], cmap="gray")
                ax.set_title("image_O_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 7:
                ax.imshow(labels_H_T1[r, :, :, 0], cmap="gray")
                ax.set_title("ground-truth_H_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 8:
                ax.imshow(images_O_T2[r, :, :, 0], cmap="gray")
                ax.set_title("image_O_T2", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 9:
                ax.imshow(labels_H_T2[r, :, :, 0], cmap="gray")
                ax.set_title("ground-truth_H_T2", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            cnt += 1

    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"data_w_labels_and_transforms{stain_code}_samples.png"),
                   bbox_inches="tight")
    plt.close(figure)


def check_plot(images_H_T1, h2o_output, labels_O_T1, images_O_T1, o2h_output, labels_H_T1, output_dir=None):
    rows = 3
    cols = 6
    figure = plt.figure(figsize=(20, 10))
    figure.subplots_adjust(wspace=0, hspace=0)
    font_size = 15
    cnt = 1
    for r in range(rows):
        for c in range(cols):
            ax = figure.add_subplot(rows, cols, cnt)
            if c == 0:
                ax.imshow(images_H_T1[r, :, :, 0], cmap="gray")
                ax.set_title("image_H_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 1:
                ax.imshow(h2o_output[r, :, :, 0], cmap="gray")
                ax.set_title("h2o_pred", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 2:
                ax.imshow(labels_O_T1[r, :, :, 0], cmap="gray")
                ax.set_title("labels_O_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 3:
                ax.imshow(images_O_T1[r, :, :, 0], cmap="gray")
                ax.set_title("images_O_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 4:
                ax.imshow(o2h_output[r, :, :, 0], cmap="gray")
                ax.set_title("o2h_pred", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            if c == 5:
                ax.imshow(labels_H_T1[r, :, :, 0], cmap="gray")
                ax.set_title("labels_H_T1", fontdict={'fontsize': font_size, 'fontweight': 'medium'})
                ax.axis("off")
            cnt += 1

    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"check.png"), bbox_inches="tight")
    plt.close(figure)
