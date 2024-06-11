# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Optional, Tuple
import logging
from iopath.common.file_io import g_pathmgr
import pickle
import json
import yaml
from multiprocessing import Pool
from functools import partial
import urllib

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from vissl_download import download_and_extract_archive


try:
    import av
except ImportError:
    raise ValueError("You must have pyav installed to run this script: pip install av.")


def save_file(data, filename, append_to_json=True, verbose=True):
    """
    Common i/o utility to handle saving data to various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    Specifically for .json, users have the option to either append (default)
    or rewrite by passing in Boolean value to append_to_json.
    """
    if verbose:
        logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "wb") as fopen:
            pickle.dump(data, fopen, pickle.HIGHEST_PROTOCOL)
    elif file_ext == ".npy":
        with g_pathmgr.open(filename, "wb") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        if append_to_json:
            with g_pathmgr.open(filename, "a") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
        else:
            with g_pathmgr.open(filename, "w") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "w") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    else:
        raise Exception(f"Saving {file_ext} is not supported yet")

    if verbose:
        logging.info(f"Saved data to file: {filename}")


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The input folder contains the expanded UCF-101 archive files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output folder containing the disk_folder output",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_const",
        const=True,
        default=False,
        help="To download the original dataset and decompress it in the input folder",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker used to decode videos",
    )
    parser.add_argument(
        "-dw",
        "--download_workers",
        type=int,
        default=1,
        help="Number of parallel worker used to download videos",
    )
    return parser


def _dl_wrap(download_root, url):
    return download_and_extract_archive(url, download_root)


def download_dataset(root: str, num_workers: int = 1):
    """
    Download the K700 dataset video path, annotations and videos
    """

    # Download the video path and the annotations
    for split in ["train", "val"]:
        download_url(
            root=root,
            url=f"https://s3.amazonaws.com/kinetics/700_2020/{split}/k700_2020_{split}_path.txt",
        )
        download_url(
            root=root,
            url=f"https://s3.amazonaws.com/kinetics/700_2020/annotations/{split}.csv",
        )

    # Download all the videos and expand the archive
    if num_workers == 1:
        for split in ["train", "val"]:
            with open(os.path.join(root, f"k700_2020_{split}_path.txt")) as f:
                for line in f:
                    video_batch_url = line.strip()
                    split_root = os.path.join(root, split)
                    download_and_extract_archive(
                        url=video_batch_url, download_root=split_root
                    )
    else:
        for split in ["train", "val"]:
            with open(os.path.join(root, f"k700_2020_{split}_path.txt")) as f:
                list_video_urls = [urllib.parse.quote(line, safe="/,:") for line in f.read().splitlines()]
                split_root = os.path.join(root, split)
                part = partial(_dl_wrap, split_root)
                poolproc = Pool(num_workers)
                poolproc.map(part, list_video_urls)



class KineticsMiddleFrameDataset:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    def __init__(self, data_path: str, split: str):
        self.data_path = data_path
        self.split = split
        self.split_path = os.path.join(data_path, split)
        self.video_paths = []
        self.video_labels = []
        self._init_dataset()

    def _init_dataset(self):
        """
        Find all the video paths and the corresponding labels
        """
        for label in os.listdir(self.split_path):
            label_path = os.path.join(self.split_path, label)
            if not os.path.isdir(label_path):
                continue

            for file_name in os.listdir(label_path):
                file_ext = os.path.splitext(file_name)[1]
                if file_ext == ".mp4":
                    self.video_paths.append(os.path.join(label_path, file_name))
                    self.video_labels.append(label)

    @staticmethod
    def _extract_middle_frame(file_path: str) -> Optional[Image.Image]:
        """
        Extract the middle frame out of a video clip following
        the protocol of CLIP (https://arxiv.org/pdf/2103.00020.pdf)
        at Appendix A.1
        """
        with av.open(file_path) as container:
            if len(container.streams.video) > 0:
                nb_frames = container.streams.video[0].frames
                vid_stream = container.streams.video[0]
                for i, frame in enumerate(container.decode(vid_stream)):
                    if i - 1 == nb_frames // 2:
                        return frame.to_image()
            return None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str, str]:
        video_path = self.video_paths[idx]
        label = self.video_labels[idx]
        mid_frame = self._extract_middle_frame(video_path)
        video_name = os.path.split(video_path)[1]
        image_name = os.path.splitext(video_name)[0] + ".jpg"
        return mid_frame, image_name, label, video_path


def clean_label(label: str) -> str:
    """
    Return a label without spaces or parenthesis
    """
    for c in "()":
        label = label.replace(c, "")
    for c in " ":
        label = label.replace(c, "_")
    return label.strip("_")


def create_split(input_path: str, output_path: str, split: str, num_workers: int):
    """
    Create one split of the disk_folder format and the associated disk_filelist files
    """
    image_paths = []
    image_labels = []
    error_paths = []

    # Create the disk_folder format
    dataset = KineticsMiddleFrameDataset(data_path=input_path, split=split)
    loader = DataLoader(
        dataset, num_workers=num_workers, batch_size=1, collate_fn=lambda x: x[0]
    )
    for mid_frame, image_name, label, video_path in tqdm(loader, total=len(dataset)):
        if mid_frame is not None:
            label = clean_label(label)
            label_folder = os.path.join(output_path, f"{split}_images", label)
            os.makedirs(label_folder, exist_ok=True)
            image_path = os.path.join(label_folder, image_name)
            with open(image_path, "w") as image_file:
                mid_frame.save(image_file)
            image_paths.append(image_path)
            image_labels.append(label)
        else:
            error_paths.append(video_path)

    # Save the disk_filelist format
    save_file(
        np.array(image_paths), filename=os.path.join(output_path, f"{split}_images.npy")
    )
    save_file(
        np.array(image_labels),
        filename=os.path.join(output_path, f"{split}_labels.npy"),
    )
    if len(error_paths):
        print(f"Number of errors in '{split}' split: {len(error_paths)}")
        error_paths_file = os.path.join(output_path, f"{split}_errors.npy")
        print(f"Errors are saved in: {error_paths_file}")
        save_file(error_paths, filename=error_paths_file)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python prepare_k700.py -i /path/to/k700 -o /output_path/k700 -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input, args.download_workers)

    for split in ["train", "val"]:
        create_split(
            input_path=args.input,
            output_path=args.output,
            split=split,
            num_workers=args.workers,
        )