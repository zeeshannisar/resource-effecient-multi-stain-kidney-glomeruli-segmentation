#!/bin/bash
#!/bin/bash

baseDir=${HOME}/phd/saved_models/UNet/SSL/unet/percentN_equally_randomly

for stain in 02 03 16 32 39;
do
  for rep in rep1 rep2 rep3 rep4 rep5;
  do
    for mode in simple;
    do
      for percent_N in percent_1 percent_5 percent_10 percent_20 percent_50 percent_100;
      do
        label=${mode}_unet_${stain}_250_${rep}
        results_txt_path=${baseDir}/${percent_N}/${stain}/${mode}_${stain}_rgb/results/${label}/results_maxoutput_${stain}.${label}.txt
        results_csv_path=${baseDir}/scores_txt2csv/${mode}.csv
        python3 results_txt2csv.py -rt ${results_txt_path} -rc ${results_csv_path} -s ${stain} -tm ${mode} -r ${rep} -pN ${percent_N}
      done
    done
  done
done



for stain in 02 03 16 32 39;
do
  for rep in rep1 rep2 rep3 rep4 rep5;
  do
    for mode in finetune;
    do
      for percent_N in percent_1 percent_5 percent_10 percent_20 percent_50 percent_100;
      do
        label=${mode}_unet_${stain}_250_${rep}
        results_txt_path=${baseDir}/${percent_N}/${stain}/${mode}_${stain}_rgb/SimCLR/Base_Scale/results/${label}/results_maxoutput_${stain}.${label}.txt
        results_csv_path=${baseDir}/scores_txt2csv/${mode}_SimCLR_Base_Scale.csv
        python3 results_txt2csv.py -rt ${results_txt_path} -rc ${results_csv_path} -s ${stain} -tm ${mode} -r ${rep} -pN ${percent_N}
      done
    done
  done
done

