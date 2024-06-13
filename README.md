## Let Go of Your Labels with Unsupervised Transfer

[Artyom Gadetsky*](http://agadetsky.github.io), [Yulun Jiang*](https://yljblues.github.io), [Maria Brbić](https://brbiclab.epfl.ch/team/)

[`Project page`](https://brbiclab.epfl.ch/projects/turtle/) | [`Paper`](https://openreview.net/pdf?id=RZHRnnGcEx) | [`BibTeX`](#citing) 
_________________
This repo contains the source code of 🐢 TURTLE, an unupervised learning algorithm written in PyTorch. 🔥 TURTLE achieves state-of-the-art unsupervised performance on the variety of benchmark datasets. For more details please check our paper [Let Go of Your Labels with Unsupervised Transfer](https://openreview.net/pdf?id=RZHRnnGcEx) (ICML '24).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-imagenet)](https://paperswithcode.com/sota/image-clustering-on-imagenet?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-cifar-100)](https://paperswithcode.com/sota/image-clustering-on-cifar-100?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-cifar-10)](https://paperswithcode.com/sota/image-clustering-on-cifar-10?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-caltech-101)](https://paperswithcode.com/sota/image-clustering-on-caltech-101?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-dtd)](https://paperswithcode.com/sota/image-clustering-on-dtd?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-flowers-102)](https://paperswithcode.com/sota/image-clustering-on-flowers-102?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-food-101)](https://paperswithcode.com/sota/image-clustering-on-food-101?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-mnist)](https://paperswithcode.com/sota/image-clustering-on-mnist?p=let-go-of-your-labels-with-unsupervised-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/let-go-of-your-labels-with-unsupervised-1/image-clustering-on-stl-10)](https://paperswithcode.com/sota/image-clustering-on-stl-10?p=let-go-of-your-labels-with-unsupervised-1)

<div align="justify">The question we aim to answer in our work is how to utilize representations from foundation models to solve a new task in a fully unsupervised manner. We introduce the problem setting of unsupervised transfer and highlight the key differences between unsupervised transfer and other types of transfer. Specifically, types of downstream transfer differ in the amount of available supervision. Given representation spaces of foundation models, (i) supervised transfer, represented as a linear probe, trains a linear classifier given labeled examples of a downstream dataset; (ii) zero-shot transfer assumes descriptions of the visual categories that appear in a downstream dataset are given, and employs them via text encoder to solve the task; and (iii) unsupervised transfer assumes the least amount of available supervision, i.e., only the number of categories is given, and aims to uncover the underlying human labeling of a dataset.</div>
</br>
<div align="center" style="padding: 0 100pt">
<img src="figures/setting_plot.png">
</div>
</br>
<div align="justify">TURTLE is a method that enables fully unsupervised transfer from foundation models. The key idea behind our approach is to search for the labeling of a downstream dataset that maximizes the margins of linear classifiers in the space of single or multiple foundation models to uncover the underlying human labeling. Compared to zero-shot and supervised transfer, unsupervised transfer with TURTLE does not need the supervision in any form. Compared to deep clustering methods, TURTLE does not require task-specific representation learning that is expensive for modern foundation models.</div>

### Dependencies
The code is built with the following libraries

- [PyTorch](https://pytorch.org/) - 2.2.1
- [torchvision](https://pytorch.org/vision/stable/index.html) - 0.17.1
- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [scikit-learn](http://scikit-learn.org)
- [clip](https://github.com/openai/CLIP)
- [tqdm](https://tqdm.github.io)
- [cuml](https://rapids.ai/) - 24.02

To install [cuml](https://rapids.ai/), you can follow the instructions on [this page](https://docs.rapids.ai/install?_gl=1*1az1x2f*_ga*MTY1NDI3MDM1MS4xNzE3NzUwMTQz*_ga_RKXFW6CM42*MTcxNzc1MDE0My4xLjAuMTcxNzc1MDE0My42MC4wLjA.).

### Quick Start
In our paper, we consider 26 vision datasets studied in [(Radford et al. 2021)](https://arxiv.org/abs/2103.00020) and 9 different foundation models. As a running example, we present the full pipeline to train TURTLE on the CIFAR100 dataset.

1. Precompute representations and save ground truth labels for the dataset
```
python precompute_representations.py --dataset cifar100 --phis clipvitL14
python precompute_representations.py --dataset cifar100 --phis dinov2 
python precompute_labels.py --dataset cifar100
```

2. Train TURTLE with 2 representation spaces
```
python run_turtle.py --dataset cifar100 --phis clipvitL14 dinov2 
```
or with the single representation space
```
python run_turtle.py --dataset cifar100 --phis clipvitL14
python run_turtle.py --dataset cifar100 --phis dinov2
```

The results and the checkpoints will be saved at ```./data/results```, ```./data/task_checkpoints```. You can also use `--root_dir` in all scripts to specify root directory instead of `./data` which is used by default.

### Data Preparation

Most datasets can be automatically downloaded by running `precompute_representations.py` and `precompute_labels.py`. However, some of the datasets require manual downloading. Please check ```dataset_preparation/data_utils.py``` for guide to prepare all the datasets used in our paper. 

As an example, to prepare `pets` dataset that is not directly available at ```torchvision.datasets```, one can run:
```
python dataset_preparation/prepare_pets.py -i ./data/datasets/pets -o ./data/datasets/pets -d
```
to download and extract the dataset at ```./data/datasets/pets```.

After downloading the dataset, run the following command to precompute the representations and labels:
```
python precompute_representations.py --dataset ${DATASET} --phis ${REPRESENTATION}
python precompute_labels.py --dataset ${DATASET}
```

Datasets and representations covered in this repo:
- 26 datasets: ```food101, cifar10, cifar100, birdsnap, sun397, cars, aircraft, dtd, pets, caltech101, flowers, mnist, fer2013, stl10, eurosat, resisc45, gtsrb, kitti, country211, pcam, ucf101, kinetics700, clevr, hatefulmemes, sst, imagenet```.
- 9 representations: ``clipRN50, clipRN101, clipRN50x4, clipRN50x16, clipRN50x64, clipvitB32, clipvitB16, clipvitL14, dinov2``.

### Running TURTLE
Once the representations and labels are precomputed, to train TURTLE with a single space, run:
```
python run_turtle.py --dataset ${DATASET} --phis ${REPRESENTATION} 
```
or to train TURTLE with multiple representation spaces, run
```
python run_turtle.py --dataset ${DATASET} --phis ${REPRESENTATION1} ${REPRESENTATION2}
```

You can also use ```--inner_lr```, ```---outer_lr```, ```--warm_start``` to specify inner step size, outer step size and whether to use cold-start or warm start bilevel optimization. Furthermore, use ``--cross_val`` to compute the generalization score for the found labeling after training. You can perform hyperparameter sweep and use the generalization score to select the best hyperparemeters **without using ground truth labels**.

### Pre-trained Checkpoints

We also release the labelings found by TURTLE for all datasets and all model architectures used in our paper. To download pre-trained checkpoints, run:
```
wget https://brbiclab.epfl.ch/wp-content/uploads/2024/06/turtle_tasks.zip
unzip turtle_tasks.zip
```
Then, you can evaluate the pre-trained checkpoint of TURTLE with the single space by running:
```
python evaluate.py --dataset cifar100 --phis clipvitL14 --task_ckpt {PATH_TO_TURTLE_TASKS}/1space/clipvitL14/cifar100.pt
python evaluate.py --dataset cifar100 --phis dinov2     --task_ckpt {PATH_TO_TURTLE_TASKS}/1space/dinov2/cifar100.pt
```
or evaluate using two representation spaces using:
```
python evaluate.py --dataset cifar100 --phis clipvitL14 dinov2 --task_ckpt {PATH_TO_TURTLE_TASKS}/2space/clipvitL14_dinov2/cifar100.pt
```

### Baselines

We also provide implemetation of *Zero-shot Transfer* with CLIP, *Linear Probe* and *K-Means* baselines in the `baselines` folder. To implement linear probe and K-Means baselines we employ [cuml](https://rapids.ai/) for highly efficient cuda implementations.

#### Linear Probe
Precompute the representations and then perform linear probe evaluation by running:
```
python baselines/linear_probe.py --dataset ${DATASET} --phis ${REPRESENTATION}
```
To select the l2 regularization strength for better performance, run 
```
python baselines/linear_probe.py --dataset ${DATASET} --phis ${REPRESENTATION} --validation
```

#### K-Means
Precompute the representations and run K-Means baseline:
```
python baselines/kmeans.py --dataset ${DATASET} --phis ${REPRESENTATION}
```

#### Zero-shot Transfer
Run CLIP zero-shot transfer:
```
python baselines/clip_zs.py --dataset ${DATASET} --phis ${REPRESENTATION}
```

### Acknowledgements

While developing TURTLE we greatly benefited from the open-source repositories:

- [HUME](https://github.com/mlbio-epfl/hume)
- [CLIP](https://github.com/openai/CLIP/tree/main)
- [SLIP](https://github.com/facebookresearch/SLIP)
- [VISSL](https://github.com/facebookresearch/vissl/tree/main)
- [DINOv2](https://github.com/facebookresearch/dinov2/tree/main)

### Citing

If you find our code useful, please consider citing:

```
@inproceedings{
    gadetsky2024let,
    title={Let Go of Your Labels with Unsupervised Transfer},
    author={Gadetsky, Artyom and Jiang, Yulun and Brbi\'c, Maria},
    booktitle={International Conference on Machine Learning},
    year={2024},
}
```
