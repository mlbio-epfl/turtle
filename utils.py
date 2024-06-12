import random

from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cluster_acc(y_pred, y_true, return_matching=False):
    """
    Calculate clustering accuracy and clustering mean per class accuracy.
    Requires scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        Accuracy in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    match = np.array(list(map(lambda i: col_ind[i], y_pred)))

    mean_per_class = [0 for i in range(D)]
    for c in range(D):
        mask = y_true == c
        mean_per_class[c] = np.mean((match[mask] == y_true[mask]))
    mean_per_class_acc = np.mean(mean_per_class)

    if return_matching:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc, match
    else:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc
    
def get_nmi(y_pred, y_true):
    """
    Calculate normalized mutual information. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        NMI in [0,1]
    """
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    return nmi

def get_ari(y_pred, y_true):
    """
    Calculate adjusted rand index. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        ARI in [0,1]
    """
    return metrics.adjusted_rand_score(y_true, y_pred)

datasets = [
    "food101",
    "cifar10",
    "cifar100",
    "birdsnap",
    "sun397",
    "cars",
    "aircraft",
    "dtd",
    "pets",
    "caltech101",
    "flowers",
    "mnist",
    "fer2013",
    "stl10",
    "eurosat",
    "resisc45",
    "gtsrb",
    "kitti",
    "country211",
    "pcam",
    "ucf101",
    "kinetics700",
    "clevr",
    "hatefulmemes",
    "sst",
    "imagenet",
    "cub"
]

datasets_to_c = {
    "food101": 101,
    "cifar10": 10,
    "cifar100": 100,
    "cifar10020": 20,
    "birdsnap": 500,
    "sun397": 397,
    "cars": 196,
    "aircraft": 100,
    "dtd": 47,
    "pets": 37,
    "caltech101": 102,
    "flowers": 102,
    "mnist": 10,
    "fer2013": 7,
    "stl10": 10,
    "eurosat": 10,
    "resisc45": 45,
    "gtsrb": 43,
    "kitti": 4,
    "country211": 211,
    "pcam": 2,
    "ucf101": 101,
    "kinetics700": 700,
    "clevr": 8,
    "hatefulmemes": 2,
    "sst": 2,
    "imagenet": 1000,
}

# food101         training set  75750, test set 25250  
# cifar10         training set  50000, test set 10000  
# cifar100        training set  50000, test set 10000  
# birdsnap        training set  37221, test set 2500   
# sun397          training set  19850, test set 19850
# cars            training set   8144, test set 8041
# aircraft        training set   6667, test set 3333
# dtd             training set   3760, test set 1880
# pets            training set   3680, test set 3669
# caltech101      training set   3060, test set 6084
# flowers         training set   2040, test set 6149
# mnist           training set  60000, test set 10000 
# fer2013         training set  28709, test set 3589   
# stl10           training set   5000, test set 8000   
# eurosat         training set  10000, test set 5000  
# resisc45        training set  25200, test set 6300
# gtsrb           training set  26640, test set 12630
# kitti           training set   5985, test set 1496
# country211      training set  42200, test set 21100
# pcam            training set 294912, test set 32768
# ucf101          training set   9537, test set 3783
# kinetics700     training set 536485, test set 33966
# clevr           training set   2000, test set 500
# hatefulmemes    training set   8500, test set 500
# sst             training set   7792, test set 1821
# imagenet        training set 1281167, test set 50000


