#!/bin/bash

cd ../../
patch_strategy=percentN_equally_randomly
pretrained_ssl_model=SimCLR
pretrained_ssl_model_name=Base_Scale
pretrained_ssl_model_at_epoch=model_epoch199.h5
pretrained_ssl_model_rep=rep1
pretrained_model_trainable=False
LR_weightdecay=None
reduceLR_percentile=None

for percent_N in percent_1;
do
  for train_mode in transferlearning;
  do
    for rep in rep2 rep3 rep4 rep5;
    do
      for test_stain in 02 03 16 32 39;
      do
        pretrained_ssl_model=SimCLR
        pretrained_ssl_model_name=Base_Scale
        pretrained_ssl_model_at_epoch=model_epoch199.h5
        pretrained_ssl_model_rep=rep1
        pretrained_model_trainable=False
        reduce_learning_rate_epoch=None
        image_dir=${HOME}/phd/data/Nephrectomies/${test_stain}/images/test_images

        python3 apply_unet.py -c configuration_files/${train_mode}_udagan.cfg -l ${train_mode}_udagan_250_${rep} -d ${image_dir} -g 0 -pm ${pretrained_ssl_model} -pmn ${pretrained_ssl_model_name} -pme ${pretrained_ssl_model_at_epoch} -pmr ${pretrained_ssl_model_rep} -ps ${patch_strategy} -pn ${percent_N} -pmt ${pretrained_model_trainable} -rle ${reduce_learning_rate_epoch}
        python3 pixel_based_evaluation.py -c configuration_files/${train_mode}_udagan.cfg -l ${train_mode}_udagan_250_${rep} -s ${test_stain} -m -g 0 -pm ${pretrained_ssl_model} -pmn ${pretrained_ssl_model_name} -pme ${pretrained_ssl_model_at_epoch} -pmr ${pretrained_ssl_model_rep} -ps ${patch_strategy} -pn ${percent_N} -pmt ${pretrained_model_trainable} -rle ${reduce_learning_rate_epoch}

      done
    done
  done
done




