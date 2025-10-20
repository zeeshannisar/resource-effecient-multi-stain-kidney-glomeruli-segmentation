import argparse
from utils import config_utils, filepath_utils
import os
import argparse
import csv
from apply_udagan import apply_model
from pixel_based_evaluation import evaluate_thresholds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a UNet model.')

    parser.add_argument('label', type=str, help='the label of the model to be tested')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()
    config = config_utils.readconfig(os.path.join(config['detector.outputpath'], 'code.' + args.label + '.cfg'))

    filePath = filepath_utils.FilepathGenerator(config)

    normalisation_filename = None
    if config['normalisation.normalise_image']:
        normalisation_filename = os.path.join(config['detector.outputpath'], 'models', 'histogram_matching_stats.hdf5')

    modelfilename = 'unet_best.' + args.label + '.hdf5'

    imageList = [f[0] for f in filePath.get_images_with_list_patients(patients=config['general.trainPatients'])]

    apply_model(imageList, config, modelfilename, args.label, normalisation_filename)

    threshold = evaluate_thresholds(filePath, os.path.join(config["segmentation.segmentationpath"], args.label),
                                    args.label, configdict['extraction.class_definitions'],
                                    config["segmentation.detectionpath"], config['general.trainPatients'],
                                    config["detector.lod"], config["general.staincode"], outputprefix="train")

    print('Best threshold: %f' % threshold)

    with open(os.path.join(config['detector.outputpath'], "models", "threshold_%s_%s.txt" % (config["general.staincode"], args.label)), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['%f' % threshold])
