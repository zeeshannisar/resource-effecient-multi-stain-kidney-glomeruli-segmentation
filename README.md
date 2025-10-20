# **Resource Efficient Multi-stain Kidney Glomeruli Segmentation via Self-Supervision**

## Highlights:

This repository contains:
> - ✅ Complete codebase for pre-training the self-supervised models (SimCLR, BYOL, HR-CS-CO) from scratch on any custom dataset.
> - ✅ Pre-trained weights for all self-supervised models, developed using our in-house renal pathology dataset (provided by Hannover Medical School). See paper for more details
> - ✅ Detailed instructions for loading and utilising the pre-trained model weights. 
> - ✅ Full code for training baseline and fine-tuned segmentation models (UNet and UDAGAN) on any external kidney glomeruli dataset, supporting reproduction of published results and generalisation study.

## Self-Supervised Pre-training:

For pre-training, the UNet encoder is used as the default backbone. However, the codebase is flexible and supports any 
state-of-the-art CNN or transformer-based deep learning architecture as a backbone. To accelerate training, distributed 
multi-GPU support is implemented using ```TensorFlow v2```. To set up the environment, install all 
dependencies with:
```
pip install -r requirements.txt
```
**Dataset Structure:**

Organize your dataset as shown below, supporting multiple domains (stains) with varying numbers of images:
```
____SSL
    ├──random_patches/colour
       ├──train
          ├──domain_1
             ├──aa.png
                  :                
             ├──zz.png
              :
              :
          ├──domain_N
             ├──aa.png
                  :                 
             ├──zz.png

       ├──validation
          ├──domain_1
             ├──aa.png
                  :                
             ├──zz.png
              :
              :
          ├──domain_N
             ├──aa.png
                  :                 
             ├──zz.png

```

**Stain Codes Reference:**

In this repository, different stain codes are used to represent various stainings in the dataset:
> - 02: PAS stain
> - 03: Jones HE
> - 32: Sirius Red
> - 16: CD68
> - 39: CD39

**Training Scripts:** 

Once the dataset has been prepared as described above, use the following scripts for pre-training:
> - **SimCLR:** ./train_simclr.sh (pre_training/ssl/simclr/slurm)
> - **BYOL:** ./train_byol.sh (pre_training/ssl/byol/slurm)
> - **HR-CS-CO:**
  >   - Run ./train_contrastive_learning.sh first.
  >   - After completion, run ./train_cross_stain_prediction.sh (both in pre_training/ssl/hr-cs-co/slurm)


## Pre-trained Model Weights:

If you prefer to bypass the pre-training step, the pre-trained model weights for all self-supervised models, 
trained on our in-house renal pathology dataset, are available for download: 
[➡️ Download Pre-trained Weights](https://seafile.unistra.fr/d/8a7fd71081644d2f86dc/). 
After downloading, use the following scripts to load and integrate them into any of the desired downstream tasks, 
such as classification or segmentation, with a particular focus on renal pathology datasets.
> - **SimCLR:** ./load_simclr_weights.sh (available in pre_training/ssl_pretrained_models/slurm)
> - **BYOL:** ./load_byol_weights.sh (available in pre_training/ssl_pretrained_models/slurm)
> - **HR-CS-CO:** ./load_hrcsco_weights.sh (available in pre_training/ssl_pretrained_models/slurm)

## Downstream Tasks Employed in the Paper / Reproducing Results:

As detailed in the paper, pre-trained weights were applied to two segmentation-based downstream tasks, each evaluated with varying proportions of labelled data (1%, 5%, 10%, and 100%). The following scripts enable the reproduction of the reported results:
> - **Kidney Glomeruli Segmentation with UNet:** 
>   - Uses labels from all stains.
>   - Training scripts for respective baseline and finetune models are available in ```downstream_tasks/unet/slurm```. For different stains, configuration files are updated with different staincode.
>   - This setup was also used for the generalisation study on public benchmark datasets (HuBMAP, KPIs).

> - **Kidney Glomeruli Segmentation using UDAGAN:** 
>   - Uses labels from only one source stain (in our experiments PAS, staincode:02).
>   - Training and fine-tuning scripts are available in ```downstream_tasks/udagan/slurm```.
