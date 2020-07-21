# -*- coding: utf-8 -*-
"""
@original_author: Biagio Brattoli
@modified_by: Yeonwoo Sung
"""

import argparse, sys

sys.path.append('JpsTraininig')
from JigsawNetwork import Network

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('model',   type=str, help='Path to pretrained model')
parser.add_argument('classes', type=int, help='Number of permutation to use')
args = parser.parse_args()


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in list(net.state_dict().items()):
        h5f.create_dataset(k, data=v.cpu().numpy())

net = Network(args.classes,groups=2)
net.load(args.model)

save_net(args.model[:-8]+'.h5',net)
