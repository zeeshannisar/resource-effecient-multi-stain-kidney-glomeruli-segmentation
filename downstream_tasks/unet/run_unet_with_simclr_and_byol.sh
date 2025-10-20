#!/bin/bash
patch_strategy=percentN_equally_randomly
pretrained_model_trainable=True
LR=0.0001
LR_weight_decay=None
reduce_LR_percentile=90
train_mode=unet_with_simclr_and_byol
epochs=250
job_label_mode=resume_train
validation_data=full # or none if dont want to use any validation data
stain=02

for rep in rep1 rep2 rep3 rep4 rep5;
do
  for percent_N in percent_1 percent_5 percent_10 percent_100;
  do
    for project in mice_and_rats kpis hubmap;
    do
      for pretrained_model in simclr;
      do
        gpu=h100
        configuration_file=${HOME}/research/code/postdoc/improve_kidney_glomeruli_segmentation/downstream_tasks/unet/${train_mode}/code/configuration_files/finetune/${project}_${stain}.cfg
        model_label=${pretrained_model,,}_finetuned_unet_${stain}_${epochs}_${rep}
        image_dir=${WORK}/research/data/additional_datasets/kidney_pathology/${project}/processed/${stain}/images/test_images
        train_mode_modified=unet_with_${pretrained_model}
        job_label=${project}_${job_label_mode}_${train_mode_modified,,}_${stain}_${percent_N}_${rep}_${gpu}
        sh submit_job.sh ${job_label} ${gpu} ${configuration_file} ${model_label} ${pretrained_model,,} ${pretrained_model_trainable} ${patch_strategy} ${percent_N} ${LR} ${LR_weight_decay} ${reduce_LR_percentile} ${validation_data} ${image_dir} ${stain} ${project} ${epochs}
      done
    done
  done
done

for rep in rep1 rep2 rep3 rep4 rep5;
do
  for percent_N in percent_1 percent_5 percent_10 percent_100;
  do
    for project in mice_and_rats kpis hubmap;
    do
      for pretrained_model in byol;
      do
        gpu=h100
        configuration_file=${HOME}/research/code/postdoc/improve_kidney_glomeruli_segmentation/downstream_tasks/unet/${train_mode}/code/configuration_files/finetune/${project}_${stain}.cfg
        model_label=${pretrained_model,,}_finetuned_unet_${stain}_${epochs}_${rep}
        image_dir=${WORK}/research/data/additional_datasets/kidney_pathology/${project}/processed/${stain}/images/test_images
        train_mode_modified=unet_with_${pretrained_model}
        job_label=${project}_${job_label_mode}_${train_mode_modified,,}_${stain}_${percent_N}_${rep}_${gpu}
        sh submit_job.sh ${job_label} ${gpu} ${configuration_file} ${model_label} ${pretrained_model,,} ${pretrained_model_trainable} ${patch_strategy} ${percent_N} ${LR} ${LR_weight_decay} ${reduce_LR_percentile} ${validation_data} ${image_dir} ${stain} ${project} ${epochs}
      done
    done
  done
done


