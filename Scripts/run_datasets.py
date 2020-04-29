#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os

from datasets.CiaoDVD import CiaoDVD
from datasets.Delicious import Delicious
from datasets.LastFM import LastFM
from datasets.Movielens import Movielens

if __name__ == "__main__":
    # Generic parameters for each dataset
    kwargs = {}
    kwargs['use_local'] = True
    kwargs['force_rebuild'] = False
    kwargs['implicit'] = False
    kwargs['save_local'] = True
    kwargs['verbose'] = False
    kwargs['split'] = False

    datasets = [CiaoDVD, Delicious, LastFM, '100K', '1M', '10M']

    for d in datasets:
        if isinstance(d, str):
            reader = Movielens(version=d, **kwargs)
        else:
            reader = d(**kwargs)
        path = os.path.join(reader.all_datasets_dir, 'stats')
        if not os.path.exists(path):
            os.makedirs(path)
        reader.describe(save_plots=True, path=path)