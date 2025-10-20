#!/bin/bash

configuration_file=../pre_training/ssl/hr-cs-co/code/configuration_files/CSCO.cfg
encoder_model=unet
repetition=rep1

cd ../code/

echo train_CSCO.py -c ${configuration_file} -m ${encoder_model} -r ${repetition}
python3 train_CSCO.py -c ${configuration_file} -m ${encoder_model} -r ${repetition}
