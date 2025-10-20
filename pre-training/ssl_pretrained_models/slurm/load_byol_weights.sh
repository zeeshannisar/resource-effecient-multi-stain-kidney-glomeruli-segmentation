#!/bin/bash
ssl_model_name="byol"
ssl_model_path="../weights/byol/byol_unet_encoder.h5"
ssl_model_trainable="True" # True for fine-tuning and False for fixed-feature setting

cd ../code
python3 load_pretrained_model_weights.py --ssl_model_name ${ssl_model_name} --ssl_model_path ${ssl_model_path} --ssl_model_trainable ${ssl_model_trainable}





