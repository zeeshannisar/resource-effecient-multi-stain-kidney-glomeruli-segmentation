import os
import glob
import argparse
from utils import config_utils, filepath_utils


def rename_model(path, srclabel, tgtlabel):
    if os.path.isdir(os.path.join(path, 'detections', srclabel)):
        os.rename(os.path.join(path, 'detections', srclabel), os.path.join(path, 'detections', tgtlabel))
    if os.path.isdir(os.path.join(path, 'segmentations', srclabel)):
        os.rename(os.path.join(path, 'segmentations', srclabel), os.path.join(path, 'segmentations', tgtlabel))
    if os.path.isdir(os.path.join(path, 'training_patches', srclabel)):
        os.rename(os.path.join(path, 'training_patches', srclabel), os.path.join(path, 'training_patches', tgtlabel))
    for f in glob.glob(os.path.join(path, 'graphs', "*." + srclabel + ".*")):
        tgtfilename = os.path.basename(f).split('.')
        tgtfilename[-2] = tgtlabel
        tgtfilename = '.'.join(tgtfilename)
        os.rename(f, os.path.join(path, 'graphs', tgtfilename))
    for f in glob.glob(os.path.join(path, 'models', "*." + srclabel + ".*")):
        tgtfilename = os.path.basename(f).split('.')
        tgtfilename[-2] = tgtlabel
        tgtfilename = '.'.join(tgtfilename)
        os.rename(f, os.path.join(path, 'models', tgtfilename))
    for f in glob.glob(os.path.join(path, "*." + srclabel + ".*")):
        tgtfilename = os.path.basename(f).split('.')
        tgtfilename[-2] = tgtlabel
        tgtfilename = '.'.join(tgtfilename)
        os.rename(f, os.path.join(path, tgtfilename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rename model.')
    parser.add_argument('label', type=str, help='the label of the model to be renamed')
    parser.add_argument('newlabel', type=str, help='the new label of the model')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    if args.label and not filepath_utils.validlabel(args.label):
        raise ValueError('The label should not contain periods "."')

    rename_model(config['detector.outputpath'], args.label, args.newlabel)
