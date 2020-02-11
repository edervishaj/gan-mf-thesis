#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import sys
import pickle
from datasets.DataReader import DataReader

class Movies(DataReader):
    """
    This class reads/downloads the Movies dataset from Kaggle and provides methods to access 3 sparse
    matrices URM_train, URM_validation, URM_test or create cross-validation folds from
    the full URM.
    """

    DATASET_NAME = 'Movies'

    dataset_name = 'rounakbanik/the-movies-dataset'
    dataset_dir = 'the-movies-dataset'
    data_file = 'ratings.csv'

    def __init__(self, **kwargs):
        """
        Constructor

        Parameters
        ----------
        kwargs:
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(Movies, self).__init__(header=True, **kwargs)
        self.process()

    def get_ratings_file(self):
        """
        Downloads the dataset
        """
        self.download_kaggle_dataset(dataset=self.dataset_name, files=self.data_file)
        self.ratings_file = os.path.join(self.all_datasets_dir, self.dataset_dir, self.data_file)


if __name__ == '__main__':
    reader = Movies(use_local=True, force_rebuild=True, implicit=True, save_local=True, verbose=True,
                     min_ratings=1, remove_top_pop=0.0)
    URM_train = reader.get_URM_train()
    URM_test = reader.get_URM_test()
    URM_valid = reader.get_URM_validation()
    # print(URM_train.nnz)
    # print(URM_test.nnz)
    # print(URM_valid.nnz)
    print(URM_train.shape)
    print(URM_test.shape)
    print(URM_valid.shape)
    # reader.describe()