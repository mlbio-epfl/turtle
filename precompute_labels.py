import argparse
import os

import torch
import numpy as np

from dataset_preparation.data_utils import get_datasets
from utils import seed_everything



def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to precompute embeddings", required=True)
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everthything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(args)


def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets
    elif hasattr(dataset, "labels"):
        return dataset.labels
    elif hasattr(dataset, "_labels"): # food101 or aircraft
        return dataset._labels
    elif hasattr(dataset, "_samples"): # cars
        return [elem[1] for elem in dataset._samples]
    #elif hasattr(dataset, "h5py"):
    #    return [dataset[i][1] for i in range(len(dataset))]
    #else:
    #    raise ValueError("Dataset does not have targets or labels")
    else:
        return [dataset[i][1] for i in range(len(dataset))]


def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)

    train_dataset, val_dataset = get_datasets(args.dataset, None, args.root_dir)
    labels_train = get_labels(train_dataset)
    labels_val = get_labels(val_dataset)
    print(f"Num train: {len(labels_train)}")
    print(f"Num val: {len(labels_val)}")
    print(f"Num classes: {len(np.unique(labels_train))}")

    labels_dir = os.path.join(args.root_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    np.save(f"{labels_dir}/{args.dataset}_train.npy", labels_train)
    np.save(f"{labels_dir}/{args.dataset}_val.npy", labels_val)


if __name__ == '__main__':
    run()