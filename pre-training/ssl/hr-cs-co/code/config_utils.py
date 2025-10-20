"""
config_utils.py: I/O script that simplify the extraction of parameters in a configuration file
"""

import configparser
import os.path


def readconfig(config_file="configuration_files/CS.cfg"):

    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file {config_file} does not exist")

    config = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file)

    configdict = {}

    configdict["config.filename"] = config_file

    # Data Parameters
    configdict["data.Dataset"] = config.get("data", "Dataset")
    configdict["data.DataDir"] = config.get("data", "DataDir")
    configdict["data.TrainDataDir"] = config.get("data", "TrainDataDir")
    configdict["data.ValidationDataDir"] = config.get("data", "ValidationDataDir")

    # Training
    configdict["training.ImageSize"] = [int(x) for x in config.get("training", "ImageSize").split(',')]
    configdict["training.CroppedImageSize"] = [int(x) for x in config.get("training", "CroppedImageSize").split(',')]
    configdict["training.ImageChannels"] = config.getint("training", "ImageChannels")
    configdict["training.BatchSize"] = config.getint("training", "BatchSize")
    configdict["training.NumEpochs"] = config.getint("training", "NumEpochs")
    configdict["training.SaveInterval"] = config.getint("training", "SaveInterval")
    configdict["training.SSLModel"] = config.get("training", "SSLModel").lower()
    configdict["training.SSLModelPhase"] = config.get("training", "SSLModelPhase").lower()
    configdict["training.MaxNumCheckpoints"] = config.get("training", "MaxNumCheckpoints")

    if configdict["training.MaxNumCheckpoints"].lower() == "none":
        configdict["training.MaxNumCheckpoints"] = None
    else:
        configdict["training.MaxNumCheckpoints"] = config.getint("training", "MaxNumCheckpoints")

    configdict["training.Optimiser"] = config.get("training", "Optimiser")
    configdict["training.LearningRate"] = config.getfloat("training", "LearningRate")

    if configdict["training.SSLModelPhase"] == "contrastive_learning":
        configdict["training.WeightDecay"] = config.getfloat("training", "WeightDecay")

    if configdict["training.SSLModelPhase"] == "cross_stain_prediction":
        configdict["training.ReduceLearningRate"] = config.getboolean("training", "ReduceLearningRate")

    configdict["training.EarlyStopping"] = config.getboolean("training", "EarlyStopping")
    configdict["training.Patience"] = config.getint("training", "Patience")

    configdict["training.DecoderLossType"] = config.get("training", "DecoderLossType")
    if configdict["training.SSLModelPhase"] == "contrastive_learning":
        configdict["training.DecoderLossWeights"] = config.getfloat("training", "DecoderLossWeights")
        configdict["training.ByolLossWeights"] = config.getfloat("training", "ByolLossWeights")

    # Stain Separation Parameters
    configdict["stain_separation.Method"] = config.get("stain_separation", "Method")
    configdict["stain_separation.Perturb_alpha_range"] = config.getfloat("stain_separation", "Perturb_alpha_range")
    configdict["stain_separation.Perturb_beta_range"] = config.getfloat("stain_separation", "Perturb_beta_range")

    # Augmentations Parameters
    configdict["augmentations.Flip"] = config.getboolean("augmentations", "Flip")
    configdict["augmentations.RandomResizedCrop"] = config.getboolean("augmentations", "RandomResizedCrop")
    configdict["augmentations.GaussianBlur"] = config.getboolean("augmentations", "GaussianBlur")

    if configdict["augmentations.RandomResizedCrop"]:
        try:
            configdict["augmentations.Scale"] = [float(x) for x in config.get("augmentations", "Scale").split(',')]
        except ValueError:
            print(f"RandomResizedCrop is set to True. Please specify the parameter Scale to file {config_file}")

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
    if configdict["augmentations.GaussianBlur"]:
        try:
            configdict["augmentations_prob.GaussianBlur_p"] = config.getfloat("augmentations_prob", "GaussianBlur_p")
        except ValueError:
            print(f"GaussianBlur is set to True. Please specify the parameter GaussianBlur_p to file {config_file}")

    # Pre-Processing Parameters
    configdict["preprocessing.Standardise"] = config.getboolean("preprocessing", "Standardise")
    configdict["preprocessing.ComputeMeanStd"] = config.getboolean("preprocessing", "ComputeMeanStd")
    configdict["preprocessing.AugmentInComputeMeanStd"] = config.getboolean("preprocessing", "AugmentInComputeMeanStd")
    configdict["preprocessing.NormalizeMeanStd"] = config.getboolean("preprocessing", "NormalizeMeanStd")

    # Output Parameters
    configdict["output.OutputDir"] = config.get("output", "OutputDir")

    return configdict
