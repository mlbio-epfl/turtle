import sys
sys.path.append('.')

import argparse
import os

import numpy as np
import cuml, cudf

from utils import seed_everything, datasets_to_c, get_cluster_acc


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for K-Means evaluation", required=True)
    parser.add_argument('--phis', type=str, default="clipvitL14", nargs='+', help="Representation spaces to run K-Means", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_init', type=int, default=10)
    return parser.parse_args(args)

def run(args=None):
    args = _parse_args(args)

    Zs_train = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_train.npy").astype(np.float32) for phi in args.phis]
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy").astype(np.float32) for phi in args.phis]
    ytrain = np.load(f"{args.root_dir}/labels/{args.dataset}_train.npy")
    yval = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")

    # For multiple representations, normalize feature from each space
    if len(args.phis) >= 2:
        Zs_train = [Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True) for Z_train in Zs_train]
        Zs_val = [Z_val / np.linalg.norm(Z_val, axis=1, keepdims=True) for Z_val in Zs_val]
 
    Ztrain, Zval = np.concatenate(Zs_train, axis=1), np.concatenate(Zs_val, axis=1)

    print('Start running KMeans!')
    Ztrain, Zval = cudf.DataFrame(Ztrain, dtype=np.float32), cudf.DataFrame(Zval, dtype=np.float32)

    kmeans = cuml.KMeans(n_clusters=datasets_to_c[args.dataset], max_iter=1000, n_init=args.n_init, verbose=1, random_state=args.seed)
    pred_train = kmeans.fit_predict(Ztrain).to_numpy()
    inertia = kmeans.inertia_
    pred_val = kmeans.predict(Zval).to_numpy()

    acc_train, _ = get_cluster_acc(pred_train, ytrain)
    acc_val, _ = get_cluster_acc(pred_val, yval)
    print(f"Train Accuracy: {acc_train * 100:.2f}")
    print(f"Val Accuracy: {acc_val * 100:.2f}")

    if not os.path.exists(f"{args.root_dir}/results"):
        os.makedirs(f"{args.root_dir}/results")

    phis = '_'.join(args.phis)
    with open(f"{args.root_dir}/results/kmeans.txt", 'a') as f:
        f.writelines(f"{args.dataset:12}, {phis:20}, n_init {args.n_init}, " +
                     f"inertia {inertia}, Train Accuracy: {acc_train * 100:.2f}, Val Accuracy: {acc_val * 100:.2f} \n")

if __name__ == '__main__':
    run()