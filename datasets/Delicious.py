#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
from datasets.DataReader import DataReader

class Delicious(DataReader):
    """
    This class reads/downloads Delicious Bookmarks dataset and provides methods to access 3 sparse
    matrices URM_train, URM_validation, URM_test or create cross-validation folds from
    the full URM.
    """

    DATASET_NAME = 'Delicious'

    url = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip'
    dataset_dir = 'hetrec2011-delicious-2k'
    data_file = 'user_taggedbookmarks-timestamps.dat'

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

        super(Delicious, self).__init__(delim='\t', header=True, **kwargs)
        self.process(split)
