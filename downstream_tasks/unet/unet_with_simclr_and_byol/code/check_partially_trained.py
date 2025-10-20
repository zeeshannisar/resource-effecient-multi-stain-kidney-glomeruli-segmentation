import os
import argparse
from utils import config_utils, filepath_utils
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check for unfinished training.')
    parser.add_argument('label', type=str, help='the label of the model to be checked')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    if args.label and not filepath_utils.validlabel(args.label):
        raise ValueError('The label should not contain periods "."')

    if os.path.isfile(os.path.join(config['detector.outputpath'], 'models', "unet_latest." + args.label + ".hdf5")):
        sys.exit(0)
    else:
        sys.exit(1)
