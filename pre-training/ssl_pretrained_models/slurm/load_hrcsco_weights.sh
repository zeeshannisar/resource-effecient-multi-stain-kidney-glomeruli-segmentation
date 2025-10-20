#!/bin/bash
# For HR-CS-CO separate model is trained for each stain therefore we need to specify the staincode in the path
ssl_model_name="hrcsco"
staincode=02
ssl_model_path="../weights/hrcsco/hrcsco_unet_encoder_${staincode}.hdf5"
ssl_model_trainable="True" # True for fine-tuning and False for fixed-feature setting

cd ../code
python3 load_pretrained_model_weights.py --ssl_model_name ${ssl_model_name} --ssl_model_path ${ssl_model_path} --ssl_model_trainable ${ssl_model_trainable}





