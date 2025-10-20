import shutil
import os
import glob
import argparse
from utils import config_utils, filepath_utils


def delete_model(filePath, path, label):
    shutil.rmtree(os.path.join(path, 'detections', label), ignore_errors=True)
    shutil.rmtree(os.path.join(path, 'segmentations', label), ignore_errors=True)
    shutil.rmtree(os.path.join(path, 'training_patches', label), ignore_errors=True)
    shutil.rmtree(filePath.get_result_path(label), ignore_errors=True)
    for f in glob.glob(os.path.join(path, 'graphs', "*." + label + ".*")):
        os.remove(f)
    for f in glob.glob(os.path.join(path, 'models', "*." + label + ".*")):
        os.remove(f)
    for f in glob.glob(os.path.join(path, "*." + label + ".*")):
        os.remove(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Delete model.')
    parser.add_argument('label', type=str, help='the label of the model to be deleted')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    if args.label and not filepath_utils.validlabel(args.label):
        raise ValueError('The label should not contain periods "."')

    filePath = filepath_utils.FilepathGenerator(config)

    delete_model(filePath, config['detector.outputpath'], args.label)
