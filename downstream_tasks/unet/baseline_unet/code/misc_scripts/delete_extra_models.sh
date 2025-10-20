#!/bin/bash

for strategy in percentN_equally_randomly percentN_randomly;
do
  for percent_N in percent_1 percent_5 percent_10 percent_20 percent_50 percent_100;
  do
    for stain in 02 03 16;
    do
      for epoch in 50 100 150 200 250;
      do
        for rep in rep1 rep2 rep3;
        do
          for mode in simple finetune;
          do
            if [ ${mode} == "simple" ];
            then
              model_path=/home/nisar/phd/saved_models/UNet/SSL/unet/conclude_best_data_strategy/${strategy}/${percent_N}/${stain}/simple_${stain}_rgb/models/unet_${epoch}.simple_unet_${stain}_250_${rep}.hdf5
            else
              model_path=/home/nisar/phd/saved_models/UNet/SSL/unet/conclude_best_data_strategy/${strategy}/${percent_N}/${stain}/finetune_${stain}_rgb/SimCLR_Base_Scale/models/unet_${epoch}.finetune_unet_${stain}_250_${rep}.hdf5
            fi

            if [ -f "${model_path}" ];
            then
              rm ${model_path}
              echo deleted: ${model_path}
            else
              echo not exist: ${model_path}
            fi
          done
        done
      done
    done
  done
done


