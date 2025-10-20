import os
import glob
import argparse
from utils import config_utils, filepath_utils


def rename_model(config, path, srclabel, tgtlabel):
    if os.path.isdir(os.path.join(path, 'detections', srclabel)):
        os.rename(os.path.join(path, 'detections', srclabel), os.path.join(path, 'detections', tgtlabel))
    if os.path.isdir(os.path.join(path, 'segmentations', srclabel)):
        os.rename(os.path.join(path, 'segmentations', srclabel), os.path.join(path, 'segmentations', tgtlabel))
    if os.path.isdir(os.path.join(path, 'training_patches', srclabel)):
        os.rename(os.path.join(path, 'training_patches', srclabel), os.path.join(path, 'training_patches', tgtlabel))

    if os.path.isdir(os.path.join(path, 'results', srclabel)):
        os.rename(os.path.join(path, 'results', srclabel), os.path.join(path, 'results', tgtlabel))
    for f in glob.glob(os.path.join(path, 'results', tgtlabel, "*." + srclabel + ".*")):
        tgtfilename = os.path.basename(f).split('.')
        tgtfilename[-2] = tgtlabel
        tgtfilename = '.'.join(tgtfilename)
        os.rename(f, os.path.join(path, 'results', tgtlabel, tgtfilename))

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


    # if os.path.isdir(os.path.join(config['detector.modelpath'], config['detector.modelname'])):
    #     os.rename(os.path.join(config['detector.modelpath'], config['detector.modelname']),
    #               os.path.join(config['detector.modelpath'], f"transferlearning_{config['general.staincode']}_{config['trainingstrategy.strategy']}"))

def pretrained_simclr_model_path(conf):
    return os.path.join(conf['general.homepath'], 'code/improve_kidney_glomeruli_segmentation/pre_training',
                        'models/SimCLR/simclr_unet_encoder.h5')

def pretrained_byol_model_path(conf):
    return os.path.join(conf['general.homepath'], 'code/improve_kidney_glomeruli_segmentation/pre_training',
                        'models/BYOL/byol_unet_encoder.h5')


def derived_parameters(conf, arguments):
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
        conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model.lower()
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable
        if 'simclr' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.simclr_model_path'] = pretrained_simclr_model_path(conf)
        elif 'byol' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.byol_model_path'] = pretrained_byol_model_path(conf)
        else:
            raise ValueError("Self-supervised learning based pretrained-models should be one of ['simclr', 'byol', 'hrcsco']")

    if conf['transferlearning.finetune']:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'],
                                                   conf['transferlearning.pretrained_ssl_model'])
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
    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, help='simclr | byol | none')
    parser.add_argument('-pmt', '--pretrained_model_trainable', default=False,
                        type=lambda x: str(x).lower() in ['true', '1', 'yes'], help='if finetune: True | if fixedfeatures: False')

    # Adding parameters to specify the data strategy
    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pn', '--percent_N', type=str, help='percent_1 | percent_5 | percent_10 | percent_20 | percent_50 | percent_100')

    parser.add_argument('-lr', '--LR', type=str, default="None")
    parser.add_argument('-lrd', '--LR_weightdecay', type=str, default="None")
    parser.add_argument('-rlrp', '--reduceLR_percentile', type=str, default="None")

    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    if args.label and not filepath_utils.validlabel(args.label):
        raise ValueError('The label should not contain periods "."')

    config = derived_parameters(config, arguments=args)
    
    if args.LR != 'None':
        config['detector.learn_rate'] = float(args.LR)

    if args.LR_weightdecay != 'None':
        config['detector.LR_weightdecay'] = float(args.LR_weightdecay)
    else:
        config['detector.LR_weightdecay'] = None

    if args.reduceLR_percentile != 'None' and config['transferlearning.finetune']:
        if config['detector.reducelr'] and config['transferlearning.pretrained_model_trainable']:
            config['detector.reduceLR_percentile'] = int(args.reduceLR_percentile)
            config['detector.reduceLR_patience'] = int((config['detector.reduceLR_percentile'] / 100) * config['detector.epochs'])

    rename_model(config, config['detector.outputpath'], args.label, args.newlabel)
