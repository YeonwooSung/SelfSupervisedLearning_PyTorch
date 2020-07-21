# -*- coding: utf-8 -*-
"""
@original_author: Biagio Brattoli
@modified_by: Yeonwoo Sung
"""

import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist


parser = argparse.ArgumentParser(description='Train network on Imagenet')
parser.add_argument('--classes', default=1000, type=int, 
                    help='Number of permutations to select')
args = parser.parse_args()


def main():
    outname = f'permutations/permutations_{args.classes}.npy'
    
    P_hat = np.array(list(itertools.permutations(list(range(9)), 9)))
    n = P_hat.shape[0]
    
    for i in trange(args.classes):
        if i==0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1,-1])
        else:
            P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
        
        P_hat = np.delete(P_hat,j,axis=0)
        D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
        j = D.argmax()
        
        if i%100==0:
            np.save(outname,P)
    
    np.save(outname,P)
    print('file created --> '+outname)


if __name__ == '__main__':
    main()
