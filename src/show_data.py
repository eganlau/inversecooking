from args import get_parser
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import random
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import get_model
from torchvision import transforms
import sys
import json
import time
import torch.backends.cudnn as cudnn
from utils.tb_visualizer import Visualizer
from model import mask_from_eos, label2onehot
from utils.metrics import softIoU, compute_metrics, update_error_types
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'




def main(args):

    # ingrs_vocab = pickle.load(open('../data/recipe1m_vocab_ingrs.pkl', 'rb'))
    # instrs_vocab = pickle.load(open('../data/recipe1m_vocab_toks.pkl', 'rb'))
    # dataset = pickle.load(open('../data/recipe1m_'+split+'.pkl', 'rb'))

    # max_num_labels = 20
    # ilabels_gt = np.ones(max_num_labels) * ingrs_vocab('<pad>')

    # print(ingrs_vocab)

    # Build data loader
    data_loaders = {}
    datasets = {}

    data_dir = args.recipe1m_dir

    for split in ['train','val']:

        transforms_list = [transforms.Resize((args.image_size))]

        if split == 'train':
            # Image preprocessing, normalization for the pretrained resnet
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            transforms_list.append(transforms.RandomCrop(args.crop_size))

        else:
            transforms_list.append(transforms.CenterCrop(args.crop_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))

        transform = transforms.Compose(transforms_list)
        max_num_samples = max(args.max_eval, args.batch_size) if split == 'val' else -1
        # print("max_num_samples: ",max_num_samples)
        data_loaders[split], datasets[split] = get_loader(data_dir, args.aux_data_dir, split,
                                                          args.maxseqlen,
                                                          args.maxnuminstrs,
                                                          args.maxnumlabels,
                                                          args.maxnumims,
                                                          transform, args.batch_size,
                                                          shuffle=split == 'train', num_workers=args.num_workers,
                                                          drop_last=True,
                                                          max_num_samples=max_num_samples,
                                                          use_lmdb=False,
                                                          suff=args.suff)
        
        print(split,": ",len(datasets[split]))

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
