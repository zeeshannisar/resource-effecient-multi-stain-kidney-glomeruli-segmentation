#!/bin/bash

configuration_file=../pre_training/ssl/simclr/code/configuration_files/simclr.cfg
encoder_model=unet
repetition=rep1

cd ../code/
python3 train_simclr_multipleGpus.py -c ${configuration_file} -m ${encoder_model} -r ${repetition}
