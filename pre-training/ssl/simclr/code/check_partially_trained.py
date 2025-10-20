import os
import argparse
import config_utils
import pathlib
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check for unfinished training.')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-rep', '--repetition', type=str, default='rep1')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    home_path = pathlib.Path.home()
    config["output.OutputDir"] = os.path.join(home_path, config["output.OutputDir"], config["data.Dataset"],
                                              config["training.SelfSupervisedModel"], config["training.DeepModel"], args.repetition,
                                              config["output.Label"])
    if os.path.isfile(os.path.join(config["output.OutputDir"], 'models', f'model_epoch{config["training.NumEpochs"]-1:03d}.h5')):
#        print("Training is finished...")
        sys.exit(0)
    else:
#        print("Training is not finished...")
        sys.exit(1)
