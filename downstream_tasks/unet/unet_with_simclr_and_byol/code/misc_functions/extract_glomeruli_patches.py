import os
import csv
import argparse
import numpy
from utils import image_utils, config_utils
import skimage.draw
import skimage.segmentation
import shutil
from PIL import Image, ImageDraw

# python3 extract_glomeruli_density_stats.py 20180408_223709 -c sysmifta_Nx_02_03_16_32_39_new.cfg

downsample_factor = 4
min_area = 8000
line_thickness = 2

def extract_glomeruli_images(glomeruliDetectionPath, imagePath, glomeruli_class_index, lod, outputpath):

    for png_patient_filename in os.listdir(os.path.join(glomeruliDetectionPath)):
        patchoutputpath = os.path.join(outputpath, os.path.splitext(png_patient_filename)[0])
        if not (os.path.exists(patchoutputpath)):
            os.mkdir(patchoutputpath)

        svs_patient_filename = os.path.splitext(png_patient_filename)[0] + '.svs'

        image_orig = image_utils.read_svs_image_forced(os.path.join(imagePath, svs_patient_filename), lod)

        glomeruli_detection = image_utils.read_image(os.path.join(glomeruliDetectionPath, png_patient_filename))
        glomeruli_detection = glomeruli_detection.copy()
        glomeruli_detection[glomeruli_detection != glomeruli_class_index] = 0

        labelmask, nbLabels = skimage.measure.label(glomeruli_detection, background=0, return_num=True)

        image = Image.fromarray(image_orig)
        image = image.resize([x//downsample_factor for x in image.size])

        draw = ImageDraw.Draw(image)

        glomeruli = {}
        glomeruli['coords'] = []
        glomeruli['areas'] = []
        glomeruli['centroid'] = []
        idx = 0
        for region in skimage.measure.regionprops(labelmask):
            # take regions with large enough areas
            glomeruli['centroid'].append([int(c) for c in region.centroid])
            glomeruli['coords'].append(region.coords)
            glomeruli['areas'].append(region.area)

            if region.area > min_area:

                for i in range(line_thickness):
                    draw.rectangle(((region.bbox[1] // downsample_factor)-i, (region.bbox[0] // downsample_factor)-i,
                                    (region.bbox[3] // downsample_factor)+i, (region.bbox[2] // downsample_factor)+i),
                                   outline='red')

                image_utils.save_image(image_orig[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]],
                                       os.path.join(patchoutputpath,
                                                    os.path.splitext(png_patient_filename)[0] + str(idx) + '.png'))

            idx += 1

        image.save(os.path.join(outputpath, os.path.splitext(png_patient_filename)[0] + '.png'))

    return glomeruli


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract glomeruli-cell density statistics.')

    parser.add_argument('label', type=str, help='the label of the model\'s output to be used')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()
    config = config_utils.readconfig(os.path.join(config['detector.outputpath'], 'sysmifta.' + args.label + '.cfg'))

    glomeruliDetectionPath = os.path.join(config['segmentation.detectionpath'], args.label, 'maxoutput')

    glomeruli_class_index = config['extraction.class_definitions']['glomeruli'][0]

    outputpath = os.path.join(config['detector.outputpath'], 'object_patches', 'glomeruli')
    print('Storing results in %s' % outputpath)
    shutil.rmtree(outputpath, ignore_errors=True)
    if not (os.path.exists(outputpath)):
        os.makedirs(outputpath)

    inputpath = os.path.join(config['general.datapath'], 'testImagesFull')
    glomeruli = extract_glomeruli_images(glomeruliDetectionPath, inputpath, glomeruli_class_index, config['detector.lod'], outputpath)

