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
import pandas as pd
from datasets.DataReader import DataReader

class BookCrossing(DataReader):
    """
    This class reads/downloads Book-Crossing dataset and provides methods to access 3 sparse
    matrices URM_train, URM_validation, URM_test or create cross-validation folds from
    the full URM.
    """

    DATASET_NAME = 'Book-Crossing'

    url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
    dataset_dir = 'BX-CSV-Dump'
    data_file = 'BX-Book-Ratings.csv'

    def __init__(self, **kwargs):

        """
        Constructor

        Parameters
        ----------
        kwargs:
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(BookCrossing, self).__init__(delim=';', header=True, **kwargs)
        self.process()


    def build_remote(self):
        """
        Downloads the Book-Crossing dataset and builds sparse matrices
        """

        zip_file = self.download_url(self.url, self.dataset_dir, self.verbose, desc='Downloading ' + self.DATASET_NAME + ' from ')
        zfile = zipfile.ZipFile(zip_file)

        try:
            self.ratings_file = zfile.extract(self.data_file, os.path.join(self.all_datasets_dir, os.path.dirname(zip_file)))

            df = pd.read_csv(self.ratings_file, sep=';', engine='c', skiprows=1, encoding='ISO-8859-1', converters={2: float})
            df = df[(df[df.columns[2]] > 0)]
            unique_isbn = dict(zip(df[df.columns[1]].values, range(0, len(df[df.columns[1]].values))))
            df[df.columns[1]] = pd.Series(df[df.columns[1]].values).map(unique_isbn).values
            df.to_csv(sep=';', encoding='utf-8', header=False, index=False, path_or_buf=self.ratings_file, quoting=3)

            self.URM = self.build_URM(file=self.ratings_file, use_cols=self.use_cols, delimiter=self.delimiter,
                                      header=self.header, save_local=self.save_local, implicit=self.implicit,
                                      remove_top_pop=self.remove_top_pop, verbose=self.verbose)

            self.URM_train, \
            self.URM_test, \
            self.URM_validation = self.split_urm(self.URM, split_ratio=self.split_ratio, save_local=self.save_local,
                                                 min_ratings=self.min_ratings, verbose=self.verbose,
                                                 save_dir=os.path.dirname(self.ratings_file))

            with open(os.path.join(os.path.dirname(self.ratings_file), 'config.pkl'), 'wb') as f:
                pickle.dump(self.config, f)

        except (FileNotFoundError, zipfile.BadZipFile):
            print('Either file ' + self.data_file + ' not found or ' + os.path.split(self.url)[-1] + ' is corrupted', file=sys.stderr)
            raise
        except AttributeError:
            print('config is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
            raise



if __name__ == '__main__':
    reader = BookCrossing(use_local=True, force_rebuild=True, implicit=True, save_local=True, verbose=True,
                     min_ratings=1, remove_top_pop=0.0)
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
    # reader.describe()