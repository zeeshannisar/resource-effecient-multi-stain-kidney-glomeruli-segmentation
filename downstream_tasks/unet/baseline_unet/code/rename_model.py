import os
import glob
import argparse
from utils import config_utils, filepath_utils


def rename_model(config, path, srclabel, tgtlabel):
    # if os.path.isdir(os.path.join(path, 'detections', srclabel)):
    #     os.rename(os.path.join(path, 'detections', srclabel), os.path.join(path, 'detections', tgtlabel))
    # if os.path.isdir(os.path.join(path, 'segmentations', srclabel)):
    #     os.rename(os.path.join(path, 'segmentations', srclabel), os.path.join(path, 'segmentations', tgtlabel))
    # if os.path.isdir(os.path.join(path, 'training_patches', srclabel)):
    #     os.rename(os.path.join(path, 'training_patches', srclabel), os.path.join(path, 'training_patches', tgtlabel))
    #
    # if os.path.isdir(os.path.join(path, 'results', srclabel)):
    #     os.rename(os.path.join(path, 'results', srclabel), os.path.join(path, 'results', tgtlabel))
    # for f in glob.glob(os.path.join(path, 'results', tgtlabel, "*." + srclabel + ".*")):
    #     tgtfilename = os.path.basename(f).split('.')
    #     tgtfilename[-2] = tgtlabel
    #     tgtfilename = '.'.join(tgtfilename)
    #     os.rename(f, os.path.join(path, 'results', tgtlabel, tgtfilename))
    #
    # for f in glob.glob(os.path.join(path, 'graphs', "*." + srclabel + ".*")):
    #     tgtfilename = os.path.basename(f).split('.')
    #     tgtfilename[-2] = tgtlabel
    #     tgtfilename = '.'.join(tgtfilename)
    #     os.rename(f, os.path.join(path, 'graphs', tgtfilename))
    # for f in glob.glob(os.path.join(path, 'models', "*." + srclabel + ".*")):
    #     tgtfilename = os.path.basename(f).split('.')
    #     tgtfilename[-2] = tgtlabel
    #     tgtfilename = '.'.join(tgtfilename)
    #     os.rename(f, os.path.join(path, 'models', tgtfilename))
    # for f in glob.glob(os.path.join(path, "*." + srclabel + ".*")):
    #     tgtfilename = os.path.basename(f).split('.')
    #     tgtfilename[-2] = tgtlabel
    #     tgtfilename = '.'.join(tgtfilename)
    #     os.rename(f, os.path.join(path, tgtfilename))


    if os.path.isdir(os.path.join(config['detector.modelpath'], config['detector.modelname'])):
        os.rename(os.path.join(config['detector.modelpath'], config['detector.modelname']),
                  os.path.join(config['detector.modelpath'], f"transferlearning_{config['general.staincode']}_{config['trainingstrategy.strategy']}"))

def pretrained_model_path(conf):
    return os.path.join(conf['general.homepath'], 'saved_models/SSL/Nephrectomy',
                        conf['transferlearning.pretrained_ssl_model'], conf['detector.segmentationmodel'],
                        conf['transferlearning.pretrained_ssl_model_rep'],
                        conf['transferlearning.pretrained_ssl_model_name'], 'models',
                        conf['transferlearning.pretrained_ssl_model_at_epoch'])


def pretrained_model_stats_file(conf):
    return os.path.join(conf['general.homepath'], 'saved_models/SSL/Nephrectomy',
                        conf['transferlearning.pretrained_ssl_model'], 'data_statistics/normalisation_stats.hdf5')


def derived_parameters(conf, arguments):
    conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model
    conf['transferlearning.pretrained_ssl_model_rep'] = arguments.pretrained_ssl_model_rep
    conf['transferlearning.pretrained_ssl_model_name'] = arguments.pretrained_ssl_model_name
    conf['transferlearning.pretrained_ssl_model_at_epoch'] = arguments.pretrained_ssl_model_at_epoch
    if conf['transferlearning.finetune']:
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable
        conf['transferlearning.pretrained_ssl_model_path'] = pretrained_model_path(conf)
        conf['transferlearning.pretrained_ssl_model_stats_file'] = pretrained_model_stats_file(conf)
    else:
        conf['transferlearning.pretrained_ssl_model_path'] = None
        conf['transferlearning.pretrained_ssl_model_stats_file'] = None

    conf['detector.patchstrategy'] = arguments.patch_strategy
    conf['detector.percentN'] = arguments.percent_N

    if conf['detector.percentN'] == "percent_100":
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'patches')
    else:
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'separated_patches',
                                                       conf['detector.patchstrategy'], conf['detector.percentN'])

    conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'])

    if conf['transferlearning.finetune']:
        if conf['transferlearning.pretrained_model_trainable']:
            conf['detector.reduce_learning_rate_epoch'] = arguments.reduce_learning_rate_epoch

        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'],
                                                   conf['transferlearning.pretrained_ssl_model'],
                                                   conf['transferlearning.pretrained_ssl_model_name'])
        conf['detector.outputpathdir'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'])
    else:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'])

    conf['segmentation.segmentationpath'] = os.path.join(conf['detector.outputpath'],
                                                         conf['segmentation.segmentationpath'])
    conf['segmentation.detectionpath'] = os.path.join(conf['detector.outputpath'],
                                                      conf['segmentation.detectionpath'])
    return conf



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rename model.')
    parser.add_argument('-l', '--label', type=str, help='the label of the model to be renamed')
    parser.add_argument('-newl', '--newlabel', type=str, help='the new label of the model')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    # Adding parameters to finetune the UNet with pretrained Self Supervised Learning Models (SimCLR, Byol, CSCO, etc)
    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, default='SimCLR')
    parser.add_argument('-pmn', '--pretrained_ssl_model_name', type=str, default='Base_Scale')
    parser.add_argument('-pme', '--pretrained_ssl_model_at_epoch', type=str, default='model_epoch199.h5')
    parser.add_argument('-pmr', '--pretrained_ssl_model_rep', type=str, default='rep1')

    # Adding parameters to specify the data strategy
    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pn', '--percent_N', type=str, default='percent_1')
    parser.add_argument('-pmt', '--pretrained_model_trainable', action="store_true", default=False)
    parser.add_argument('-rle', '--reduce_learning_rate_epoch', type=int, default=10)
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    if args.label and not filepath_utils.validlabel(args.label):
        raise ValueError('The label should not contain periods "."')

    config = derived_parameters(config, arguments=args)

    rename_model(config, config['detector.outputpath'], args.label, args.newlabel)
