import os
import argparse
import shutil
import json
import random
from tqdm import tqdm
import numpy as np


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/mnt/data", help='Root dir to store everthything')
    return parser.parse_args(args)


def run(args=None):
	args = _parse_args(args)
	dataset_path = os.path.join(args.root_dir, "datasets/CLEVR_v1.0")
	if os.path.exists(os.path.join(dataset_path, "train")):
		print("Train dir already exists, but we recompute!")
		shutil.rmtree(os.path.join(dataset_path, "train"))
	os.makedirs(os.path.join(dataset_path, "train"))
	for count in ['10', '3', '4', '5', '6', '7', '8', '9']:
		os.makedirs(os.path.join(dataset_path, "train", count))

	if os.path.exists(os.path.join(dataset_path, "test")):
		print("Test dir already exists, but we recompute!")
		shutil.rmtree(os.path.join(dataset_path, "test"))
	os.makedirs(os.path.join(dataset_path, "test"))
	for count in ['10', '3', '4', '5', '6', '7', '8', '9']:
		os.makedirs(os.path.join(dataset_path, "test", count))

	with open(os.path.join(dataset_path, "scenes", "CLEVR_train_scenes.json"), "r") as f:
		metadata = json.load(f)

	random.seed(42)
	selected_scenes = random.sample(metadata["scenes"], 2500)

	for scene in selected_scenes[:2000]:
		curr_count = str(len(scene["objects"]))
		shutil.copyfile(
			os.path.join(dataset_path, "images/train", scene["image_filename"]),
			os.path.join(dataset_path, "train", curr_count, scene["image_filename"])
		)

	for scene in selected_scenes[2000:]:
		curr_count = str(len(scene["objects"]))
		shutil.copyfile(
			os.path.join(dataset_path, "images/train", scene["image_filename"]),
			os.path.join(dataset_path, "test", curr_count, scene["image_filename"])
		)


if __name__ == '__main__':
    run()
