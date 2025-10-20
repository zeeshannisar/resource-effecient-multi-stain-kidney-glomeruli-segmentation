#!/bin/bash

patch_strategy=percentN_equally_randomly
pretrained_model_trainable=False
LR=0.0001
LR_weight_decay=None
reduce_LR_percentile=None
train_mode=fixedfeatures
validation_data=full # or none if dont want to use any validation data
stain=02
for pretrained_ssl_model in byol simclr;
do
  for rep in rep1 rep2 rep3 rep4 rep5;
  do
    for percent_N in percent_1 percent_5 percent_10 percent_100;
    do
      configuration_file=${HOME}/phd/code/improve_kidney_glomeruli_segmentation/downstream_tasks/udagan/code/configuration_files/${train_mode}/sysmifta.cfg
      model_label=${train_mode,,}_${pretrained_ssl_model,,}_udagan_${stain}_250_${rep}
      cd ../code/
      python3 train_udagan.py -c ${configuration_file} -l ${model_label} -pm ${pretrained_ssl_model,,} -pmt ${pretrained_model_trainable} -ps ${patch_strategy} -pn ${percent_N}  -lr ${LR} -lrd ${LR_weight_decay} -rlrp ${reduce_LR_percentile} -vd ${validation_data}
    done
  done
done
