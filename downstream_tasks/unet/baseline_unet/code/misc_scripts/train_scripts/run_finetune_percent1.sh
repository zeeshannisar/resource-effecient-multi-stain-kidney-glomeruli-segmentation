#!/bin/bash

cd ../../
patch_strategy=percentN_equally_randomly
pretrained_ssl_model=SimCLR
pretrained_ssl_model_name=Base_Scale
pretrained_ssl_model_at_epoch=model_epoch199.h5
pretrained_ssl_model_rep=rep1
pretrained_model_trainable=True

#np.logspace(np.log10(0.0001), np.log10(0.1), num=5, endpoint=True, dtype=None, axis=0)
logspace_lr=(0.0001 0.00056234 0.00316228 0.01778279 0.1)
weightdecay=0.0001
reduceLR_percentile=70

#for lr in ${logspace_lr[@]};
for LR in 0.0001;# 0.00056234 0.00316228 0.01778279 0.1;
do
  for percent_N in percent_1;
  do
    for stain in 02;# 03 16 32 39;
    do
      for rep in rep1:
      do
          python3 train_unet.py -c configuration_files/finetune_with_SimCLR_${stain}.cfg -l finetune_unet_${stain}_250_${rep} -pn ${percent_N} -g 0 -pm ${pretrained_ssl_model} -pmn ${pretrained_ssl_model_name} -pme ${pretrained_ssl_model_at_epoch} -pmr ${pretrained_ssl_model_rep} -ps ${patch_strategy} -pmt ${pretrained_model_trainable} -lr ${LR} -lrd ${weightdecay} -rlrp ${reduceLR_percentile}
      done
    done
  done
done

