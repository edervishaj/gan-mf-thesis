#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import sys
import pickle
import zipfile
from datasets.DataReader import DataReader

class CiaoDVD(DataReader):
    """
    This class reads/downloads CiaoDVD dataset and provides methods to access 3 sparse
    matrices URM_train, URM_validation, URM_test or create cross-validation folds from
    the full URM.
    """

    url = 'https://www.librec.net/datasets/CiaoDVD.zip'
    dataset_dir = 'CiaoDVD'
    data_file = 'movie-ratings.txt'

    def __init__(self, **kwargs):

        """
        Constructor

        Parameters
        ----------
        kwargs:
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(CiaoDVD, self).__init__(**kwargs)
        self.process()


if __name__ == '__main__':
    reader = CiaoDVD(use_local=True, force_rebuild=True, implicit=True, save_local=True, verbose=True, min_ratings=1, remove_top_pop=0.1)
    URM_train = reader.get_URM_train()
    URM_test = reader.get_URM_test()
    URM_valid = reader.get_URM_validation()
    # print(URM_train.nnz)
    # print(URM_test.nnz)
    # print(URM_valid.nnz)
    print(URM_train.shape)
    print(URM_test.shape)
    print(URM_valid.shape)
    reader.describe()

    # for train, test in reader.get_CV_folds(verbose=False):
    #     print(train.nnz)
    #     print(test.nnz)