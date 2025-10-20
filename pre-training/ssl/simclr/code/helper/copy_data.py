import os
import shutil
from tqdm import tqdm
import random
from datetime import date
import argparse
import json


def get_stain_names(stain):
    switcher = {
        "02": "PAS",
        "03": "Jones_HE",
        "16": "CD68",
        "32": "Sirius_Red",
        "39": "CD34"
    }
    return switcher.get(stain, "")


class CopyData:
    def __init__(self, stains, nb_images_to_copy, mode, from_copy_dir, to_copy_dir):
        self.stains = stains
        self.nb_images_to_copy = nb_images_to_copy
        self.mode = mode
        self.from_copy_dir = from_copy_dir
        self.to_copy_dir = to_copy_dir

    def copy_data(self):

        for stain in self.stains:
            from_copy_dir = os.path.join(self.from_copy_dir, stain, "patches", "colour", self.mode, "images", "images")
            to_copy_dir = os.path.join(self.to_copy_dir, "patches", "colour", self.mode, "images", stain)
            os.makedirs(to_copy_dir, exist_ok=True)

            random.seed(date.today().year)
            random_files = random.sample(os.listdir(from_copy_dir), self.nb_images_to_copy)
            self.random_copy(stain, random_files, from_copy_dir, to_copy_dir)

    def random_copy(self, stain, random_files, from_copy_dir, to_copy_dir):
        for file in (pbar := tqdm(random_files)):
            pbar.set_description(f"Copying {len(random_files)} patches from {self.mode} {get_stain_names(stain)}...")
            src = os.path.join(from_copy_dir, file)
            dst = os.path.join(to_copy_dir, file)
            shutil.copy(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copying Data...')
    parser.add_argument("-n", "--nb_images_to_copy", type=int, default=50)
    parser.add_argument("-m", "--mode", type=str, default="train")

    import pathlib
    home_path = pathlib.Path.home()
    args = parser.parse_args()

    args.from_copy_dir = f"{home_path}/phd/data/Nephrectomies_random"
    args.to_copy_dir = f"{home_path}/phd/data/self_supervised/Nephrectomies_random"
    args.stains = ["02", "03", "16", "32", "39"]

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    datacopy = CopyData(args.stains, args.nb_images_to_copy, args.mode, args.from_copy_dir, args.to_copy_dir)
    datacopy.copy_data()