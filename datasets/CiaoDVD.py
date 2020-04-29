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
    
    DATASET_NAME = 'CiaoDVD'


    url = 'https://www.librec.net/datasets/CiaoDVD.zip'
    dataset_dir = 'CiaoDVD'
    data_file = 'movie-ratings.txt'

    def __init__(self, split=True, **kwargs):

        """
        Constructor

        Parameters
        ----------
        split: bool, default True
            Flag that indicates whether to split the full URM into train, test and validation URMs

        kwargs:
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(CiaoDVD, self).__init__(**kwargs, use_cols={'user_id':0, 'item_id':1, 'rating':4})
        self.process(split)
