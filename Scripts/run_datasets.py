#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

from datasets.BookCrossing import BookCrossing
from datasets.CiaoDVD import CiaoDVD
from datasets.Delicious import Delicious
from datasets.LastFM import LastFM
from datasets.Movielens import Movielens

if __name__ == "__main__":
    reader = BookCrossing(use_local=True, force_rebuild=True, implicit=True, save_local=True, verbose=True,
                     min_ratings=1, remove_top_pop=0.0, header=True)
    URM = reader.get_URM_full()
    URM_train = reader.get_URM_train()
    URM_test = reader.get_URM_test()
    URM_valid = reader.get_URM_validation()
    # print(URM_train.nnz)
    # print(URM_test.nnz)
    # print(URM_valid.nnz)
    print(URM.shape)
    print(URM_train.shape)
    print(URM_test.shape)
    print(URM_valid.shape)
    reader.describe()