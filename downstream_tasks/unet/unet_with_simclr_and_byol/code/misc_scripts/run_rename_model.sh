#!/bin/bash

for percent_N in percent_1 percent_5 percent_10 percent_20 percent_50 percent_100;
do
  for stain in 16 32 39;
  do
    for rep in rep1;
    do
      python3 rename_model.py -c configuration_files/transferlearning_with_SimCLR_${stain}.cfg -l finetune_unet_${stain}_250_${rep} -newl transferlearning_unet_${stain}_250_${rep} -pn ${percent_N}
    done
  done
done

