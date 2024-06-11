import argparse
import os
from tqdm import tqdm

import torch
from torch.nn import functional as F
import numpy as np

import clip
from dataset_preparation.data_utils import get_dataloaders, _convert_image_to_rgb, _safe_to_tensor

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to precompute embeddings")
    parser.add_argument('--phis', type=str, default="clipvitL14", help="Representation spaces to run TURTLE", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everthything')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args(args)


def get_features(dataloader, model, device):
    all_features = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            features = model(x.to(device))
            all_features.append(features.detach().cpu())

    return torch.cat(all_features).numpy()


phi_to_name = {'clipRN50': 'RN50', 'clipRN101': 'RN101', 'clipRN50x4': 'RN50x4', 'clipRN50x16': 'RN50x16', 'clipRN50x64': 'RN50x64',
                   'clipvitB32': 'ViT-B/32', 'clipvitB16': 'ViT-B/16', 'clipvitL14': 'ViT-L/14'}

def run(args=None):
    args = _parse_args(args)
    device = torch.device(args.device)

    if args.phis == 'dinov2':
        torch.hub.set_dir(os.path.join(args.root_dir, "checkpoints/dinov2"))
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        preprocess = None
    else:
        ckpt_dir = os.path.join(args.root_dir, "checkpoints/clip")
        model, preprocess = clip.load(phi_to_name[args.phis], device=device, download_root=ckpt_dir)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        model = model.encode_image
        preprocess.transforms[2] = _convert_image_to_rgb
        preprocess.transforms[3] = _safe_to_tensor
    
    trainloader, valloader = get_dataloaders(args.dataset, preprocess, args.batch_size, args.root_dir)
    feats_train = get_features(trainloader, model, device)
    feats_val = get_features(valloader, model, device)

    representations_dir = f"{args.root_dir}/representations/{args.phis}"
    if not os.path.exists(representations_dir):
        os.makedirs(representations_dir)

    np.save(f'{representations_dir}/{args.dataset}_train.npy', feats_train)
    np.save(f'{representations_dir}/{args.dataset}_val.npy', feats_val)

if __name__ == '__main__':
    run()