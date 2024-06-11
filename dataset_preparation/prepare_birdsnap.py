import os
import argparse
import shutil
import random
from torchvision.datasets.folder import pil_loader


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/mnt/data", help='Root dir to store everthything')
    return parser.parse_args(args)


def run(args=None):
	args = _parse_args(args)
	dataset_path = os.path.join(args.root_dir, "datasets/birdsnap")
	if os.path.exists(os.path.join(dataset_path, "train")):
		print("Train dir already exists, but we recompute!")
		shutil.rmtree(os.path.join(dataset_path, "train"))
	shutil.copytree(os.path.join(dataset_path, "download/images"), os.path.join(dataset_path, "train"))

	if os.path.exists(os.path.join(dataset_path, "test")):
		print("Test dir already exists, but we recompute!")
		shutil.rmtree(os.path.join(dataset_path, "test"))
	os.makedirs(os.path.join(dataset_path, "test"))

	random.seed(0)
	for subdir in os.listdir(os.path.join(dataset_path, "train")):
		os.makedirs(os.path.join(dataset_path, "test", subdir))
		all_images = os.listdir(os.path.join(dataset_path, "train", subdir))
		all_images = list(filter(lambda x: x.split(".")[1] != "gif", all_images))
		all_images_good = []
		for file in all_images:
			# some images might be corrupted i.e. Mourning_Dove/181010.jpg
			try:
				img = pil_loader(os.path.join(dataset_path, "train", subdir, file))
				all_images_good.append(file)
			except:
				print(f"Corrupted image {os.path.join(subdir, file)}")
				os.remove(os.path.join(dataset_path, "train", subdir, file))
		test_images = random.sample(all_images_good, 5)

		#test_images = random.sample(all_images, 5)
		assert len(test_images) == len(set(test_images))
		for file in test_images:
			shutil.move(
				os.path.join(dataset_path, "train", subdir, file),
				os.path.join(dataset_path, "test", subdir, file)
			)


if __name__ == '__main__':
    run()
