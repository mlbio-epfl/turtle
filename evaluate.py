import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import datasets_to_c, get_cluster_acc


def _parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to evaluate TURTLE")
    parser.add_argument('--phis', type=str, default="clipvitL14", nargs='+', help="Representation spaces to run TURTLE", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--root_dir', type=str, default='data', help='Root dir to store everything')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint to evaluate")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = _parse_args()

    # Load pre-computed representations 
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy").astype(np.float32) for phi in args.phis]
    y_gt_val = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")

    print(f'Load dataset {args.dataset}')
    print(f'Representations of {args.phis}: ' + ' '.join(str(Z_val.shape) for Z_val in Zs_val))

    C = datasets_to_c[args.dataset]
    feature_dims = [Z_val.shape[1] for Z_val in Zs_val]
    
    # Task encoder
    task_encoder = [nn.Linear(d, C).to(args.device) for d in feature_dims] 
    ckpt = torch.load(args.ckpt_path)
    for task_phi, ckpt_phi in zip(task_encoder, ckpt.values()):
        task_phi.load_state_dict(ckpt_phi)

    # Evaluate clustering accuracy
    label_per_space = [F.softmax(task_phi(torch.from_numpy(Z_val).to(args.device)), dim=1) for task_phi, Z_val in zip(task_encoder, Zs_val)]
    labels = sum(label_per_space) / len(label_per_space) # shape of (N, K)

    y_pred = labels.argmax(dim=-1).detach().cpu().numpy()
    cluster_acc, _ = get_cluster_acc(y_pred, y_gt_val)
    
    phis = '_'.join(args.phis)
    print(f'{args.dataset:12}, {phis:20}, Number of found clusters {len(np.unique(y_pred))}, Cluster Acc: {cluster_acc:.4f}')