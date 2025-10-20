import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import argparse

import config_utils

import numpy as np
import tensorflow as tf

from tf_dataloader_API import TFDataLoader
from train_CS import LoadDatasetMeanStd, normalise, ExtractCentralRegion, GetInputShape, GetOutputShape
from deepmodels.subclass_API import HO_UnetDecoder, HO_UnetEncoder, UnetEncoder, UnetDecoder
from tensorboard_API import TensorboardLogs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cross Stain Prediction...!")
    parser.add_argument('-c', '--configFile', type=str, default="configuration_files/CS.cfg", help='cfg file to use')
    parser.add_argument('-m', '--deepModel', type=str, default="UNet", help='network architecture to use')
    parser.add_argument("-r", "--repetition", type=str, default="rep1")
    parser.add_argument("-mode", "--mode", type=str, default="best")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("\n")
    print("Gpu Growth Restriction Done...")
    print("\n")

    args = parser.parse_args()

    if args.configFile:
        config = config_utils.readconfig(args.configFile)
    else:
        config = config_utils.readconfig()

    print('Command Line Input Arguments:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    print("\n")

    if args.deepModel.lower() == "unet" or args.deepModel.lower() == "resnet50":
        config["training.DeepModel"] = args.deepModel.lower()
    else:
        raise ValueError("Please specify one of (\'UNet, ResNet50\') in parser.parse_args().DeepModel.")

    if "rep" in args.repetition:
        config["training.repetition"] = args.repetition
    else:
        raise ValueError("Please specify proper repetition in parser.parse_args().repetition.")

    config["output.DataStatsDir"] = os.path.join(config["output.OutputDir"], config["training.DeepModel"],
                                                 config["output.Label"])
    config["output.OutputDir"] = os.path.join(config["output.OutputDir"], config["training.DeepModel"],
                                              config["output.Label"], config["training.repetition"],
                                              config["training.SSLModelPhase"])

    config["training.TrainNumExamples"] = TFDataLoader(config, ssl_model_phase=config["training.SSLModelPhase"],
                                                       mode="train").GetPNGImageFiles(num_examples_mode=True)

    config["data.mean"], config["data.stddev"] = LoadDatasetMeanStd(tmp_config=config, savePath="data_statistics")
    print("\nConfiguration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))

    input_shape = GetInputShape(config)
    print(f"input_shape: {input_shape}")

    encoder, decoder = UnetEncoder(), UnetDecoder()
    combined_encoder, combined_decoder = HO_UnetEncoder(), HO_UnetDecoder()
    output_shape = GetOutputShape(combined_encoder, combined_decoder, input_shape)
    print(f"output_shape: {output_shape}")

    tensorboard = TensorboardLogs(config=config)

    combined_encoder.load_weights(f"{config['output.OutputDir']}/models/HO_encoder_model.{args.mode}.hdf5")
    combined_decoder.load_weights(f"{config['output.OutputDir']}/models/HO_decoder_model.{args.mode}.hdf5")

    combined_encoder.trainable, combined_decoder.trainable = False, False

    ds_valid = TFDataLoader(config, ssl_model_phase=config["training.SSLModelPhase"],
                            mode="validation", print_data_specifications=False).LoadDataset()
    for step, inputs in enumerate(ds_valid.take(1)):
        _, image_H, label_O, image_O, label_H = inputs
        image_H = tf.image.flip_left_right(image_H)
        label_O = tf.image.flip_left_right(label_O)

        if config["preprocessing.NormalizeMeanStd"]:
            image_H = normalise(image_H, config["data.mean"], config["data.mean"])
            image_O = normalise(image_O, config["data.mean"], config["data.mean"])

        label_H = np.array([ExtractCentralRegion(x, shape=output_shape[:-1]) for x in label_H])
        label_O = np.array([ExtractCentralRegion(x, shape=output_shape[:-1]) for x in label_O])

        h_encoder_output, o_encoder_output, h_skip0, h_skip1, h_skip2, h_skip3, \
        o_skip0, o_skip1, o_skip2, o_skip3 = combined_encoder(image_H, image_O, training=False)

        h2o_decoder_output, o2h_decoder_output = combined_decoder(h_encoder_output, o_encoder_output,
                                                                  h_skip0, h_skip1, h_skip2, h_skip3,
                                                                  o_skip0, o_skip1, o_skip2, o_skip3,
                                                                  training=False)
        tensorboard.save_to_dir(image_H, image_O, label_H, label_O, h2o_decoder_output, o2h_decoder_output, f"flip_test.{args.mode}")



