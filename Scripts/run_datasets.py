#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os

from datasets.BookCrossing import BookCrossing
from datasets.CiaoDVD import CiaoDVD
from datasets.Delicious import Delicious
from datasets.Jester import Jester
from datasets.LastFM import LastFM
from datasets.Movielens import Movielens
from datasets.Movies import Movies

if __name__ == "__main__":
    # Generic parameters for each dataset
    kwargs = {}
    kwargs['use_local'] = True
    kwargs['force_rebuild'] = False
    kwargs['implicit'] = False
    kwargs['save_local'] = True
    kwargs['verbose'] = False
    kwargs['split'] = False

    datasets = [BookCrossing, CiaoDVD, Delicious, Jester, LastFM, '1M', '10M', 'latest']
    # datasets = [BookCrossing]

    for d in datasets:
        if isinstance(d, str):
            reader = Movielens(version=d, **kwargs)
        else:
            reader = d(**kwargs)
        path = os.path.join(reader.all_datasets_dir, 'stats')
        if not os.path.exists(path):
            os.makedirs(path)
        reader.describe(save_plots=True, path=path)