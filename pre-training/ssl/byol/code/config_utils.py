"""
config_utils.py: I/O script that simplify the extraction of parameters in a configuration file
"""

import configparser
import os.path


def readconfig(config_file="SimCLR_Base.cfg"):

    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file {config_file} does not exist")

    config = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file)

    configdict = {}

    configdict["config.filename"] = config_file

    # Data Parameters
    configdict["data.Dataset"] = config.get("data", "Dataset")
    configdict["data.DataDir"] = config.get("data", "DataDir")
    configdict["data.Mode"] = config.get("data", "Mode")

    # Training
    configdict["training.ImageSize"] = [int(x) for x in config.get("training", "ImageSize").split(',')]
    configdict["training.CroppedImageSize"] = [int(x) for x in config.get("training", "CroppedImageSize").split(',')]
    configdict["training.ImageChannels"] = config.getint("training", "ImageChannels")
    configdict["training.MaximumBatchSize"] = config.getint("training", "MaximumBatchSize")
    configdict["training.BatchSize"] = config.getint("training", "BatchSize")
    configdict["training.NumEpochs"] = config.getint("training", "NumEpochs")
    configdict["training.SaveInterval"] = config.getint("training", "SaveInterval")
    configdict["training.DeepModel"] = config.get("training", "DeepModel")
    configdict["training.SelfSupervisedModel"] = config.get("training", "SelfSupervisedModel").lower()
    configdict["training.HiddenLayers"] = [int(x) for x in config.get("training", "HiddenLayers").split(',')]
    configdict["training.TargetBetaDecay"] = config.getfloat("training", "TargetBetaDecay")
    configdict["training.Optimizer"] = config.get("training", "Optimizer")
    configdict["training.Momentum"] = config.getfloat("training", "Momentum")
    configdict["training.LR"] = config.getfloat("training", "LR")
    configdict["training.LRScaling"] = config.get("training", "LRScaling")
    configdict["training.WarmupEpochs"] = config.getfloat("training", "WarmupEpochs")
    configdict["training.WeightDecay"] = config.getfloat("training", "WeightDecay")
    configdict["training.MaxNumCheckpoints"] = config.get("training", "MaxNumCheckpoints")

    if configdict["training.MaxNumCheckpoints"].lower() == "none":
        configdict["training.MaxNumCheckpoints"] = None
    else:
        configdict["training.MaxNumCheckpoints"] = config.getint("training", "MaxNumCheckpoints")

    # Preprocessing Parameters
    configdict["preprocessing.Standardise"] = config.getboolean("preprocessing", "Standardise")
    configdict["preprocessing.ComputeMeanStd"] = config.getboolean("preprocessing", "ComputeMeanStd")
    configdict["preprocessing.Normalise"] = config.getboolean("preprocessing", "Normalise")

    # Augmentations Parameters
    configdict["augmentations.RandomResizedCrop"] = config.getboolean("augmentations", "RandomResizedCrop")
    configdict["augmentations.Flip"] = config.getboolean("augmentations", "Flip")
    configdict["augmentations.ColorJitter"] = config.getboolean("augmentations", "ColorJitter")
    configdict["augmentations.RGB2GRAY"] = config.getboolean("augmentations", "RGB2GRAY")
    configdict["augmentations.GaussianBlur"] = config.getboolean("augmentations", "GaussianBlur")
    configdict["augmentations.Solarize"] = config.getboolean("augmentations", "Solarize")
    configdict["augmentations.GridDistort"] = config.getboolean("augmentations", "GridDistort")
    configdict["augmentations.GridShuffle"] = config.getboolean("augmentations", "GridShuffle")

    if configdict["augmentations.GaussianBlur"]:
        try:
            configdict["augmentations.GaussianBlurDivider"] = config.getfloat("augmentations", "GaussianBlurDivider")
        except ValueError:
            print(f"GaussianBlur is set to True. Please specify the parameter GaussianBlurDivider to file {config_file}")

    # Augmentations Probability Parameters
    if configdict["augmentations.RandomResizedCrop"]:
        try:
            configdict["augmentations_prob.RandomResizedCrop_p"] = config.getfloat("augmentations_prob",
                                                                                   "RandomResizedCrop_p")
        except ValueError:
            print(f"RandomResizedCrop is set to True. Please specify the parameter RandomResizedCrop_p "
                  f"to file {config_file}")

    if configdict["augmentations.Flip"]:
        try:
            configdict["augmentations_prob.Flip_p"] = config.getfloat("augmentations_prob", "Flip_p")
        except ValueError:
            print(f"Flip is set to True. Please specify the parameter Flip_p to file {config_file}")

    if configdict["augmentations.ColorJitter"]:
        try:
            configdict["augmentations_prob.ColorJitter_p"] = config.getfloat("augmentations_prob", "ColorJitter_p")
        except ValueError:
            print(f"ColorJitter is set to True. Please specify the parameter ColorJitter_p to file {config_file}")

    if configdict["augmentations.RGB2GRAY"]:
        try:
            configdict["augmentations_prob.RGB2GRAY_p"] = config.getfloat("augmentations_prob", "RGB2GRAY_p")
        except ValueError:
            print(f"RGB2GRAY is set to True. Please specify the parameter RGB2GRAY_p to file {config_file}")

    if configdict["augmentations.GaussianBlur"]:
        try:
            configdict["augmentations_prob.GaussianBlur_p"] = config.getfloat("augmentations_prob", "GaussianBlur_p")
        except ValueError:
            print(f"GaussianBlur is set to True. Please specify the parameter GaussianBlur_p to file {config_file}")

    if configdict["augmentations.Solarize"]:
        try:
            configdict["augmentations_prob.Solarize_p"] = config.getfloat("augmentations_prob", "Solarize_p")
        except ValueError:
            print(f"GaussianBlur is set to True. Please specify the parameter GaussianBlur_p to file {config_file}")

    if configdict["augmentations.GridDistort"]:
        try:
            configdict["augmentations_prob.GridDistort_p"] = config.getfloat("augmentations_prob", "GridDistort_p")
        except ValueError:
            print(f"GridDistort is set to True. Please specify the parameter GridDistort_p to file {config_file}")

    if configdict["augmentations.GridShuffle"]:
        try:
            configdict["augmentations_prob.GridShuffle_p"] = config.getfloat("augmentations_prob", "GridShuffle_p")
        except ValueError:
            print(f"GridShuffle is set to True. Please specify the parameter GridShuffle_p to file {config_file}")

    # Output Parameters
    configdict["output.OutputDir"] = config.get("output", "OutputDir")

    return configdict
