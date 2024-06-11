import os
import argparse
import shutil
import random

from tqdm import tqdm
import numpy as np
import pandas as pd


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/mnt/data", help='Root dir to store everthything')
    return parser.parse_args(args)


def run(args=None):
	args = _parse_args(args)
	dataset_path = os.path.join(args.root_dir, "datasets/hatefulmemes")
	if os.path.exists(os.path.join(dataset_path, "train")):
		print("Train dir already exists, but we recompute!")
		shutil.rmtree(os.path.join(dataset_path, "train"))
	os.makedirs(os.path.join(dataset_path, "train"))
	for count in ['0', '1']:
		os.makedirs(os.path.join(dataset_path, "train", count))

	if os.path.exists(os.path.join(dataset_path, "test")):
		print("Test dir already exists, but we recompute!")
		shutil.rmtree(os.path.join(dataset_path, "test"))
	os.makedirs(os.path.join(dataset_path, "test"))
	for count in ['0', '1']:
		os.makedirs(os.path.join(dataset_path, "test", count))

	with open(os.path.join(dataset_path, "train.jsonl"), "r") as f:
		metadata_train = pd.read_json(path_or_buf=f, lines=True)

	with open(os.path.join(dataset_path, "dev.jsonl"), "r") as f:
		metadata_val = pd.read_json(path_or_buf=f, lines=True)

	for i, row in metadata_train.iterrows():
		shutil.copyfile(
			os.path.join(dataset_path, row["img"]),
			os.path.join(dataset_path, "train", str(row["label"]), row["img"].split("/")[1])
		)

	for i, row in metadata_val.iterrows():
		shutil.copyfile(
			os.path.join(dataset_path, row["img"]),
			os.path.join(dataset_path, "test", str(row["label"]), row["img"].split("/")[1])
		)


if __name__ == '__main__':
    run()
