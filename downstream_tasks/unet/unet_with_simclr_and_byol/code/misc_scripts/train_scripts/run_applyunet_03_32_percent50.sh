#!/bin/bash

percent_N=percent_50
learning_rate=0p0001
reduced_learning_rate_epoch=10
reduce_by_factor=0p1
rep=rep1


for stain in 03;# 32;
do
  python3 apply_unet.py -c configuration_files/finetune_with_SimCLR_${stain}.cfg -l finetune_unet_${stain}_250_LR-${learning_rate}_RF-${reduce_by_factor}_AE-${reduced_learning_rate_epoch}_${rep} -d /home/nisar/phd/data/Nephrectomies/${stain}/images/test_images -g 1 -pn ${percent_N} -pmt
  python3 pixel_based_evaluation.py -c configuration_files/finetune_with_SimCLR_${stain}.cfg -l finetune_unet_${stain}_250_LR-${learning_rate}_RF-${reduce_by_factor}_AE-${reduced_learning_rate_epoch}_${rep} -s ${stain} -m -g 1 -pn ${percent_N} -pmt;
done




