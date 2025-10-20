# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageDraw, ImageChops
import numpy
import argparse
from utils import config_utils, filepath_utils
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstance, ImageInstanceCollection
import getpass
import string
import logging
import warnings


def get_masks_per_project(cytomine_host, cytomine_public_key, cytomine_private_key, project_id, object_id,
                          object_label, stain_code, lod, file_path):

    with Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path='/api/', working_path='/tmp/',
                    verbose=logging.ERROR):

        image_instances = ImageInstanceCollection().fetch_with_filter("project", project_id)

        print("Number of images in project %d" % len(image_instances))
        scale_factor = 1.0 / (2 ** lod)

        for image in image_instances:

            image_id = image.id
            image_name = os.path.splitext(image.instanceFilename)[0]

            try:
                im_stain_code = file_path.get_stain(image_name)
            except IndexError:
                continue

            if im_stain_code == stain_code:
                output_filename = file_path.generate_maskpath(image_name, object_label, lod)

                print("Image: %s" % image_name)

                get_object_mask_per_image(cytomine_host, cytomine_public_key, cytomine_private_key, image_id,
                                          project_id, object_id, scale_factor, output_filename)


def get_object_mask_per_image(cytomine_host, cytomine_public_key, cytomine_private_key, image_id, project_id, object_id,
                              scale_factor, output_filename):

    with Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path='/api/', working_path='/tmp/',
                    verbose=logging.ERROR):

        image_instances = ImageInstanceCollection().fetch_with_filter("project", project_id)
        image_instance = [im for im in image_instances if im.id == image_id]
        image_instance = image_instance[0]

        image_width = int(image_instance.width * scale_factor)
        image_height = int(image_instance.height * scale_factor)

        annotations = AnnotationCollection()
        annotations.project = project_id
        annotations.image = image_id
        annotations.fetch()

        result_image = Image.new(mode='1', size=(image_width, image_height), color=0)

        somethingannotated = False
        for annotation in annotations:  # for each annotation (i.e. polygone)
            annotation.fetch()

            if not annotation.term:
                #raise ValueError("Annotation %d has not been associated with a term" % annotation.id)
                warnings.warn("Annotation %d has not been associated with a term" % annotation.id)

            # Get the polygon coordinates from cytomine
            if annotation.term and annotation.term[0] == int(object_id):  #  if the object is the one we want to extract

                if annotation.location.startswith("POLYGON") or annotation.location.startswith("MULTIPOLYGON"):

                    somethingannotated = True

                    if annotation.location.startswith("POLYGON"):
                        label = "POLYGON"
                    elif annotation.location.startswith("MULTIPOLYGON"):
                        label = "MULTIPOLYGON"

                    coordinatesStringList = annotation.location.replace(label, '')

                    if label == "POLYGON":
                        coordinates_string_lists = [coordinatesStringList]
                    elif label == "MULTIPOLYGON":
                        coordinates_string_lists = coordinatesStringList.split(')), ((')

                        coordinates_string_lists = [coordinatesStringList.replace('(', '').replace(')', '') for
                                                    coordinatesStringList in coordinates_string_lists]

                    for coordinatesStringList in coordinates_string_lists:
                        #  create lists of x and y coordinates
                        x_coords = []
                        y_coords = []
                        for point in coordinatesStringList.split(','):
                            point = point.strip(string.whitespace)  # remove leading and ending spaces
                            point = point.strip(string.punctuation) # Have seen some strings have a ')' at the end so remove it
                            x_coords.append(round(float(point.split(' ')[0])))
                            y_coords.append(round(float(point.split(' ')[1])))

                        x_coords_correct_lod = [int(x * scale_factor) for x in x_coords]
                        y_coords_correct_lod = [image_height - int(x * scale_factor) for x in y_coords]
                        coords = [(i, j) for i, j in zip(x_coords_correct_lod, y_coords_correct_lod)]

                        #  draw the polygone in an image and fill it
                        ImageDraw.Draw(result_image).polygon(coords, outline=1, fill=1)

        if somethingannotated:
            result_image.save(output_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration filename to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    file_path = filepath_utils.FilepathGenerator(config)

    print("Cytomine public and private keys can be found in the Account page after login")
    try:
        cytomine_public_key = raw_input("Public Key: ")
    except NameError:
        pass
        cytomine_public_key = input("Public Key: ")
    cytomine_private_key = getpass.getpass("Private Key: ")

    for i in range(len(config['extraction.objectLabels'])):

        object_label = config['extraction.objectLabels'][i]
        object_id = config['extraction.objectIds'][i]

        if not object_id == "-1":
            print("--------" + object_label + "--------")

            get_masks_per_project(config['extraction.cytominehost'], cytomine_public_key, cytomine_private_key,
                                  config['extraction.projectId'], object_id, object_label, config['general.staincode'],
                                  config['general.lod'], file_path)

            if config['general.lod'] != config['detector.lod']:
                get_masks_per_project(config['extraction.cytominehost'], cytomine_public_key, cytomine_private_key,
                                      config['extraction.projectId'], object_id, object_label,
                                      config['general.staincode'], config['detector.lod'], file_path)
