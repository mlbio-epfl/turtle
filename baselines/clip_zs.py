import sys
sys.path.append('.')

from collections import defaultdict
import argparse
import os

import torch
import torch.nn.functional as F
import clip
import numpy as np
from tqdm import tqdm

from zs_templates import datasets_to_classes, datasets_to_templates
from utils import seed_everything
from dataset_preparation.data_utils import get_dataloaders, _convert_image_to_rgb, _safe_to_tensor

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for zero-shot evaluation", required=True)
    parser.add_argument('--phis', type=str, default="clipvitL14", choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14'])
    parser.add_argument('--default_template_only', dest='default_template', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(args)


def get_zeroshot_weights(model, classnames, templates):
    """
        https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if isinstance(classname, list):
                texts = [template.format(c) for template in templates for c in classname]
            else:
                texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classify(model, zs_weights, dataloader, device):
    """
        https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    """
    all_probs = []
    all_targets = []
    with torch.no_grad():
        perclass = defaultdict(int)
        perclass_n = defaultdict(int)
        for i, (images, target) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            target = target.numpy()
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zs_weights
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(target)
            # measure accuracy
            preds = logits.argmax(1).detach().cpu().numpy()
            unq, cnt = np.unique(target, return_counts=True)
            for j, u in enumerate(unq):
                perclass[u] += (target[target == u] == preds[target == u]).sum()
                perclass_n[u] += cnt[j]

    stats = {
        "top1": sum(perclass.values()) / sum(perclass_n.values()),
        "per_class": sum([perclass[k] / perclass_n[k] for k in perclass.keys()]) / len(perclass),
        "probs": np.concatenate(all_probs),
        "targets": np.concatenate(all_targets)
    }
    return stats


phi_to_name = {'clipRN50': 'RN50', 'clipRN101': 'RN101', 'clipRN50x4': 'RN50x4', 'clipRN50x16': 'RN50x16', 'clipRN50x64': 'RN50x64',
                   'clipvitB32': 'ViT-B/32', 'clipvitB16': 'ViT-B/16', 'clipvitL14': 'ViT-L/14'}

def run(args=None):
    args = _parse_args(args)
    device = torch.device(args.device)
    seed_everything(args.seed)

    ckpt_dir = os.path.join(args.root_dir, "checkpoints/clip")
    model, preprocess = clip.load(phi_to_name[args.phis], device=device, download_root=ckpt_dir)
    model.eval()
    preprocess.transforms[2] = _convert_image_to_rgb
    preprocess.transforms[3] = _safe_to_tensor
    _, valloader = get_dataloaders(args.dataset, preprocess, args.batch_size, args.root_dir)

    if args.default_template:
        weights = get_zeroshot_weights(model, datasets_to_classes[args.dataset], ["A photo of a {}."])
    else:
        weights = get_zeroshot_weights(model, datasets_to_classes[args.dataset], datasets_to_templates[args.dataset])
    
    stats = zeroshot_classify(model, weights, valloader, device)
    print(f"Top1 Accuracy: {stats['top1'] * 100:.2f}")
    print(f"Mean Per Class Accuracy: {stats['per_class'] * 100:.2f}")

    if not os.path.exists(f"{args.root_dir}/results"):
        os.makedirs(f"{args.root_dir}/results")
    
    with open(f"{args.root_dir}/results/zeroshot.txt", 'a') as f:
        f.writelines(f"{args.dataset:12s}, {args.phis:10}, Top1 Accuracy {stats['top1'] * 100:.2f}, Mean Per Class Accuracy: {stats['per_class'] * 100:.2f} \n")

if __name__ == '__main__':
    run()