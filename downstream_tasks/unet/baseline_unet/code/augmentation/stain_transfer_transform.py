import numpy
import glob
import os
import time
from utils import image_utils
import random
import staintools
from .stain_transform import transform as stain_variance_transform


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


def transform(image, mask, data_dir, stainsubdir, target_stain_codes, transform_stain_variance=False, alpha_range=0., beta_range=0.):

    dtype = image.dtype

    image = image.astype(numpy.uint8)

    # pick a stain code
    target_stain = random.choice(target_stain_codes)

    # find all files
    files = []
    for subdir in sorted(os.listdir(os.path.join(data_dir, target_stain, 'downsampledpatches', stainsubdir, 'images'))):
        if subdir != 'background':
            files += [os.path.join(data_dir, target_stain, 'downsampledpatches', stainsubdir, 'images', subdir, name) for name in
                      os.listdir(os.path.join(data_dir, target_stain, 'downsampledpatches', stainsubdir, 'images', subdir))]

    target_image_filename = random.choice(files)

    target_image = image_utils.read_image(target_image_filename)

    if transform_stain_variance:
        target_image, _ = stain_variance_transform(target_image, mask, target_stain, alpha_range=alpha_range, beta_range=beta_range)

    try:
        #stain_normalizer = staintools.StainNormalizer(method='vahadane') # Very slow!
        stain_normalizer = staintools.StainNormalizer(method='macenko')
        stain_normalizer.fit(target_image)
        image = stain_normalizer.transform(image).astype(dtype)
    except:
        pass


    return image, mask
