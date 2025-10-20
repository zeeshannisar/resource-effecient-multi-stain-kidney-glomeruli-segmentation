import os
import json
import argparse
import shutil
import math

import numpy as np

from tqdm import tqdm


def size(data_mode, num_images, class_name, files):
    if data_mode == "validation":
        if class_name == "glomeruli":
            return num_images
        if class_name == "background":
            return math.ceil(0.08 * num_images)
        if class_name == "negative":
            return math.ceil(7 * num_images)
    else:
        return len(files)


def copy(stain, class_name, files, src_images_dir, src_gts_dir, dst_images_dir, dst_gts_dir):
    for file in tqdm(files, desc=f"copying files of stain: {stain} --- class: {class_name}..."):
        shutil.copy(src=os.path.join(src_images_dir, file), dst=os.path.join(dst_images_dir, file))
        shutil.copy(src=os.path.join(src_gts_dir, file), dst=os.path.join(dst_gts_dir, file))





def mix(to_mix_stains, base_dir, patch_strategy, percent_N, data_mode, num_images):
    np.random.seed(2023)
    for stain in to_mix_stains:
        if data_mode == "train":
            images_dir = os.path.join(base_dir, stain, "separated_patches", patch_strategy, percent_N, data_mode, "images")
            gts_dir = os.path.join(base_dir, stain, "separated_patches", patch_strategy, percent_N, data_mode, "gts")
            class_labels_json_file = os.path.join(base_dir, stain, "separated_patches", patch_strategy, percent_N, data_mode, "class_labels.json")
        else:
            images_dir = os.path.join(base_dir, stain, "patches", data_mode, "images")
            gts_dir = os.path.join(base_dir, stain, "patches", data_mode, "gts")
            class_labels_json_file = os.path.join(base_dir, stain, "patches", data_mode, "class_labels.json")

        class_names = [x[1] for x in os.walk(images_dir)][0]
        for class_name in class_names:
            files = os.listdir(os.path.join(images_dir, class_name))
            files = np.random.choice(files, size=size(data_mode, num_images, class_name, files), replace=False)

            src_images_dir = os.path.join(images_dir, class_name)
            src_gts_dir = os.path.join(gts_dir, class_name)
            dst_images_dir = os.path.join(base_dir, "mixed", "separated_patches", patch_strategy, percent_N, data_mode, "images", class_name)
            dst_gts_dir = os.path.join(base_dir, "mixed", "separated_patches", patch_strategy, percent_N, data_mode, "gts", class_name)

            os.makedirs(dst_images_dir, exist_ok=True)
            os.makedirs(dst_gts_dir, exist_ok=True)

            copy(stain, class_name, files, src_images_dir, src_gts_dir, dst_images_dir, dst_gts_dir)


    shutil.copy(src=class_labels_json_file, dst=os.path.join(base_dir, "mixed", "separated_patches", patch_strategy, percent_N, data_mode, "class_labels.json"))








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Umap plot for already trained unet for all stains...')

    parser.add_argument('-s', '--to_mix_stains', type=lambda s: [str(item) for item in s.split(',')], default="02,03,16,32,39")
    parser.add_argument('-bd', '--base_dir', type=str, default="/home/nisar/phd/data/Nephrectomies")
    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pN', '--percent_N', type=str, default="percent_1")
    parser.add_argument('-dm', '--data_mode', type=str, default="validation")
    parser.add_argument('-N', '--num_images', type=int, default=100)

    args = parser.parse_args()
    print('Command Line Input Arguments:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    print("\n")

    mix(args.to_mix_stains, args.base_dir, args.patch_strategy, args.percent_N, args.data_mode, args.num_images)



