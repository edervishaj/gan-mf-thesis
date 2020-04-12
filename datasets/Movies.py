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

    def __init__(self, split=True,**kwargs):
        """
        Constructor

        Parameters
        ----------
        kwargs:
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(Movies, self).__init__(header=True, **kwargs)
        self.process(split)

    def get_ratings_file(self):
        """
        Downloads the dataset
        """
        self.download_kaggle_dataset(dataset=self.dataset_name, files=self.data_file)
        self.ratings_file = os.path.join(self.all_datasets_dir, self.dataset_dir, self.data_file)