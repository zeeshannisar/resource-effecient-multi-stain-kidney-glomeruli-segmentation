import os
import json
import pyvips
from glob import glob
import numpy
import argparse
import tensorflow as tf
from cycgan import CycleGAN_models
from augmentation.live_augmentation import load_img
from utils.image_utils import save_image


def load_image2image_translation_model(target_stain=None):
    model = CycleGAN_models.CycleGAN(stain=target_stain)
    model.combined.load_weights(f'/home/nisar/phd/saved_models/cycleGAN/colour_transfer_models/rep1/02/{target_stain}/combined.h5')
    return model.g_AB

def preprocess_image(image):
    return (image / 127.5) - 1


def read_image(path):
    return load_img(path, grayscale=False, target_size=(508, 508), data_format=tf.keras.backend.image_data_format())


def PAS_to_Others(base_path, data_mode, image_path, image_name, class_name, translate_PAS_to_Other_direction=None):
    image = preprocess_image(image=read_image(path=image_path))

    if translate_PAS_to_Other_direction == "02_to_03":
        translated_image = (PAS_to_JonesHE.predict(image[numpy.newaxis, ...]) + 1) * 127.5
    elif translate_PAS_to_Other_direction == "02_to_16":
        translated_image = (PAS_to_CD68.predict(image[numpy.newaxis, ...]) + 1) * 127.5
    elif translate_PAS_to_Other_direction == "02_to_32":
        translated_image = (PAS_to_SiriusRed.predict(image[numpy.newaxis, ...]) + 1) * 127.5
    elif translate_PAS_to_Other_direction == "02_to_39":
        translated_image = (PAS_to_CD34.predict(image[numpy.newaxis, ...]) + 1) * 127.5
    else:
        raise ValueError("Not a valid direction to translate PAS_to_Others. translate_PAS_to_Other_direction should "
                         "be one of ['02_to_03', '02_to_16', '02_to_32', '02_to_39']")

    save_dir = os.path.join(base_path.rsplit("/", 1)[0], "translations", "PAS_to_Others", data_mode,
                            translate_PAS_to_Other_direction, class_name)
    os.makedirs(save_dir, exist_ok=True)
    save_image(translated_image[0, :, :, :], savePath=os.path.join(save_dir, image_name))


def translate(base_path, data_mode):
    images_path = glob(os.path.join(base_path, data_mode, "images", "*", "*.png"))
    print(f"Found {len(images_path)} images belonging to {len(os.listdir(os.path.join(base_path, data_mode)))} classes")

    for count, image_path in enumerate(images_path):
        stain_code = base_path.rsplit("/", 2)[1]
        image_name = image_path.rsplit("/", 2)[2]
        class_name = image_path.rsplit("/", 2)[1]

        print(f"Processing image_file: {count+1} / {len(images_path)} --- "
              f"stain_code: {stain_code} --- class_name: {class_name}", end="\r", flush=True)
        if stain_code == "02":
            PAS_to_Others(base_path, data_mode, image_path, image_name, class_name, "02_to_03")
            PAS_to_Others(base_path, data_mode, image_path, image_name, class_name, "02_to_16")
            PAS_to_Others(base_path, data_mode, image_path, image_name, class_name, "02_to_32")
            PAS_to_Others(base_path, data_mode, image_path, image_name, class_name, "02_to_39")

    print("****************** Done...! ******************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate PAS to Others...")
    parser.add_argument('-bd', '--base_path', type=str, default="/home/nisar/phd/data/Nephrectomies/02/patches")
    parser.add_argument('-dm', '--data_mode', type=str, default="train")
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')

    args = parser.parse_args()

    print('Command Line Input Arguments:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    print("\n")

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Selected GPU : " + os.environ["CUDA_VISIBLE_DEVICES"])

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("\nGpu Growth Restriction Done...")

    PAS_to_JonesHE = load_image2image_translation_model(target_stain="03")
    PAS_to_CD68 = load_image2image_translation_model(target_stain="16")
    PAS_to_SiriusRed = load_image2image_translation_model(target_stain="32")
    PAS_to_CD34 = load_image2image_translation_model(target_stain="39")

    translate(base_path=args.base_path, data_mode=args.data_mode)



