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
    delicious_dir = 'hetrec2011-delicious-2k'

    #TODO: Fix the data_file, no user-bookmark pairs
    data_file = 'user_taggedbookmarks-timestamps.dat'

    def __init__(self, **kwargs):
        """
        Constructor

        Parameters
        ----------
        kwargs:
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(Delicious, self).__init__(delim='\t', header=True, use_cols={'user_id':0, 'item_id':1}, **kwargs)
        self.process(os.path.join(self.delicious_dir, self.data_file))


if __name__ == '__main__':
    reader = Delicious(use_local=True, force_rebuild=True, implicit=True, save_local=True, verbose=True,
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
    reader.describe()