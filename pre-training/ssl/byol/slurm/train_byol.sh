#!/bin/bash

configuration_file=../pre_training/ssl/byol/code/configuration_files/byol.cfg
encoder_model=unet
repetition=rep1

cd ../code/
python3 train_byol_multipleGPUs.py -c ${configuration_file} -m ${encoder_model} -r ${repetition}
