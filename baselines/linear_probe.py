import argparse
import os

import numpy as np
import cuml, cudf
from sklearn.model_selection import train_test_split

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for Linear Probe evaluation", required=True)
    parser.add_argument('--phis', type=str, default="clipvitL14", help="Representation spaces to run Linear Probe", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--validation', action='store_true', help='Select regularization strength via cross-validation')
    parser.add_argument('--Cs', type=int, default=96)
    parser.add_argument('--Crange', type=int, default=6)
    return parser.parse_args(args)


def run(args=None):
    args = _parse_args(args)

    Ztrain = np.load(f"{args.root_dir}/representations/{args.phis}/{args.dataset}_train.npy").astype(np.float32)
    Zval = np.load(f"{args.root_dir}/representations/{args.phis}/{args.dataset}_val.npy").astype(np.float32)
    ytrain = np.load(f"{args.root_dir}/labels/{args.dataset}_train.npy")
    yval = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")

    print(Ztrain.shape, ytrain.shape)
    print(len(np.unique(ytrain)))
    if not args.validation:
        # Just use default value of "C"        
        Ztrain, ytrain = cudf.DataFrame(Ztrain), cudf.Series(ytrain)
        Zval, yval = cudf.DataFrame(Zval), cudf.Series(yval)

        clf = cuml.LogisticRegression(verbose=1, tol=1e-8, C=1.)
        clf.fit(Ztrain, ytrain)

        train_acc = clf.score(Ztrain, ytrain)
        val_acc = clf.score(Zval, yval)

        best_C = 1.
    else:
        # validation part
        Ztrain_cv, Zval_cv, ytrain_cv, yval_cv = train_test_split(Ztrain, ytrain, test_size=0.2)
        
        Ztrain_cv, ytrain_cv = cudf.DataFrame(Ztrain_cv), cudf.Series(ytrain_cv)
        Zval_cv, yval_cv = cudf.DataFrame(Zval_cv), cudf.Series(yval_cv)

        Cs = np.logspace(-args.Crange, args.Crange, args.Cs)
        validation_acc = []
        for i, C in enumerate(Cs):
            clf = cuml.LogisticRegression(verbose=1, C=C, tol=1e-8)

            clf.fit(Ztrain_cv, ytrain_cv)
            acc = clf.score(Zval_cv, yval_cv)
            validation_acc.append(acc)
            print(f'Fold {i}/{args.Cs} C={C}, validation accuracy {acc * 100:.2f}')

        best_C = Cs[np.argmax(validation_acc)]
        
        # training with best `C` from validation
        clf = cuml.LogisticRegression(verbose=1, C=best_C, tol=1e-8)
        clf.fit(Ztrain, ytrain)

        train_acc = clf.score(Ztrain, ytrain)
        val_acc = clf.score(Zval, yval)

    print(f"Train Accuracy: {train_acc * 100:.2f}")
    print(f"Val Accuracy: {val_acc * 100:.2f}")

    if not os.path.exists(f"{args.root_dir}/results"):
        os.makedirs(f"{args.root_dir}/results")

    with open(f"{args.root_dir}/results/linear_probe.txt", 'a') as f:
        f.writelines(f"{args.dataset:12s}, {args.phis}, validation {args.validation}, C={best_C}, Train Accuracy: {train_acc * 100:.2f}, Val Accuracy: {val_acc * 100:.2f} \n")

if __name__ == '__main__':
    run()