import numpy
import glob
import os
import time
from utils import image_utils,data_utils
import random
from .stain_transform import transform as stain_variance_transform
import tensorflow.keras.backend as K
import tensorflow as tf
import  cv2
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = numpy.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def crop_image(img, target_size, data_format):
    if data_format == 'channels_first':
        img_size = img.shape[1:]
    elif data_format:
        img_size = img.shape[:-1]

    if img_size[0] < target_size[0] or img_size[1] < target_size[1]:
        raise ValueError('Invalid cropped image size (%s). Image is %d x %d and target size is %d x %d.' % (
            path, img_size[0], img_size[1], target_size[0], target_size[1]))

    if (img_size[0] - target_size[0]) % 2 != 0:
        raise ValueError(
            'Invalid cropped image size. There should be an even difference between the image and target heights')

    if (img_size[1] - target_size[1]) % 2 != 0:
        raise ValueError(
            'Invalid cropped image size. There should be an even difference between the image and target widths')

    if img_size != target_size:
        diffs = numpy.subtract(img_size, target_size)
        diffs //= 2

        img = img[diffs[0]:img_size[0] - diffs[0], diffs[1]:img_size[1] - diffs[1]]

    return img


def load_img(path, grayscale=False, target_size=None, data_format=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    img = pil_image.open(path)

    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')

    img = img_to_array(img, data_format=data_format)

    if target_size is not None:
        img = crop_image(img, target_size, data_format=data_format)

    return img


def generate_from_directory(classname, numberofsamples, datapath, outputpath, stainsubdir, target_stain_codes, changefilename=True):

    os.makedirs(os.path.join(outputpath, 'images', classname), exist_ok=True)
    os.makedirs(os.path.join(outputpath, 'gts', classname), exist_ok=True)

    filenames = glob.glob(os.path.join(datapath, 'images', classname, "*.png"))

    # Get random indexes for the number of images to generate
    idx = numpy.random.randint(len(filenames), size=numberofsamples)

    for c, ind in enumerate(idx):
        if changefilename:
            save_filename = os.path.splitext(os.path.basename(filenames[ind]))[0] + '_' + str(c) + '.png'
        else:
            save_filename = os.path.basename(filenames[ind])

        image = image_utils.read_image(filenames[ind])
        maskfilename = os.path.join(datapath, 'gts', classname, os.path.splitext(os.path.basename(filenames[ind]))[0] + '.png')
        mask = image_utils.read_image(maskfilename)

        img_result, msk_result = transform(image, mask, stainsubdir, target_stain_codes)

        image_utils.save_image(img_result.astype('uint8'), os.path.join(outputpath, 'images', classname, save_filename))

        image_utils.save_image(msk_result.astype('uint8'), os.path.join(outputpath, 'gts', classname, save_filename))


def transform(image, mask, target_stain_funs, transform_stain_variance=False, alpha_range=0., beta_range=0.):


    image = image.astype(numpy.uint8)

    # pick a stain code
    target_stain_codes = list(target_stain_funs.keys())
    target_stain = random.choice(target_stain_codes)

    image = (image / 127.5) - 1 # [-1,1]
    orig_dims = image.shape
    image = take_central_path_of_shape(image,(508,508))

    image = numpy.squeeze(target_stain_funs[target_stain](image[numpy.newaxis, ...]))

    image = (image + 1) * 127.5 # [0-255]

    if transform_stain_variance:
        target_stain = target_stain.split("_")[0]
        image, _ = stain_variance_transform(image, mask, target_stain, alpha_range=alpha_range, beta_range=beta_range)

    image = resize(image,orig_dims[0],orig_dims[1])
    return image, mask


def already_saved(image, mask, target_stain_choice, translation_dir, image_directory, filename,
                  transform_stain_variance=False, alpha_range=0., beta_range=0.):
    orig_dims = image.shape
    # pick a random translation
    translation_choice = random.choice(target_stain_choice)
    image = load_img(os.path.join(translation_dir, image_directory.rsplit("/", 1)[1], translation_choice, filename))


    if transform_stain_variance:
        target_stain = translation_choice.rsplit("_", 1)[1]
        image, _ = stain_variance_transform(image, mask, target_stain, alpha_range=alpha_range, beta_range=beta_range)

    image = resize(image, orig_dims[0], orig_dims[1])
    return image, mask


def take_central_path_of_shape(img,shape):
    x = shape[0]
    y = shape[1]
    x_off = (img.shape[0]-x)//2
    y_off = (img.shape[1] - y) // 2
    return img[x_off:img.shape[0]-x_off, y_off:img.shape[1]-y_off]

def resize(img,H,W):
    return cv2.resize(img,(H,W),cv2.INTER_LINEAR)