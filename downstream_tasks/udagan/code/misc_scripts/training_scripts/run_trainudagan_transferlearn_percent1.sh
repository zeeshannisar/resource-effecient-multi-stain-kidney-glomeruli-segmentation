#!/bin/bash

cd ../../

patch_strategy=percentN_equally_randomly
pretrained_ssl_model=BYOL
pretrained_ssl_model_name=Base
pretrained_ssl_model_at_epoch=best_model_weights.h5
pretrained_ssl_model_rep=rep1
pretrained_model_trainable=False
LR_weightdecay=None
reduceLR_percentile=None

for percent_N in percent_1;
do
  for train_mode in transferlearning;
  do
    for rep in rep1;
    do
      python3 train_unet.py -c configuration_files/${train_mode}_udagan.cfg -l ${train_mode}_udagan_250_${rep} -g 0 -pm ${pretrained_ssl_model} -pmn ${pretrained_ssl_model_name} -pme ${pretrained_ssl_model_at_epoch} -pmr ${pretrained_ssl_model_rep} -ps ${patch_strategy} -pn ${percent_N} -pmt ${pretrained_model_trainable} -lrd ${LR_weightdecay} -rlrp ${reduceLR_percentile}
    done
  done
done
