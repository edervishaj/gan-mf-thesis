#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import csv
import array
import zipfile
import numpy as np
import pandas as pd
from datasets.DataReader import DataReader

class Jester(DataReader):
    """
    This class reads/downloads Jester dataset and provides methods to access 3 sparse
    matrices URM_train, URM_validation, URM_test or create cross-validation folds from
    the full URM.
    """

    DATASET_NAME = 'Jester'

    url = 'http://eigentaste.berkeley.edu/dataset/'
    download_files = ['jester_dataset_1_1.zip', 'jester_dataset_1_2.zip', 'jester_dataset_1_3.zip']
    extract_files = ['jester-data-1.xls', 'jester-data-2.xls', 'jester-data-3.xls']
    dataset_dir = 'jester'
    data_file = 'ratings.csv'

    def __init__(self, **kwargs):
        """
        Constructor

        Parameters
        ----------
        kwargs:
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(Jester, self).__init__(delim=',', **kwargs)
        self.process()

    def get_ratings_file(self):
        """
        Downloads the dataset.
        Also preprocessing is required to convert from Excel format to CSV, so prepare rows, cols and data
        in one reading for faster processing.
        """

        curr_start = 0
        d = {'row': array.array('I'), 'col': array.array('I'), 'data': array.array('f')}
        for i, part in enumerate(self.download_files):
            tmp = self.download_url(self.url+part, self.verbose,
                        desc='Downloading Jester dataset part'+str(i+1) + ' from ' + self.url+part)
            extracted = zipfile.ZipFile(tmp).extract(self.extract_files[i], os.path.join(self.all_datasets_dir, os.path.dirname(tmp)))
            os.remove(tmp)
            df = pd.read_excel(extracted, header=None)
            os.remove(extracted)

            # The first element is the number of rated jokes, so we drop it
            df.drop(df.columns[0], axis=1, inplace=True)

            # Jester data range is [-10, 10] with 99 meaning `not rated`. We add 11 to all data to shift it [1, 21]
            # and replace values 99+11=110 with 0 in order to sparsify the dataframe
            df = df + 11.0
            df.replace(110.0, 0, inplace=True)
            df = df.astype(float).round(2) # For some reason, read_excel adds precision problems

            next_start = curr_start + len(df)
            df['col'] = df.apply(lambda row: [i for i, x in zip(range(df.shape[1]), row) if x != 0], axis=1)
            df['data'] = df[df.columns[:-1]].apply(lambda row: [x for x in row if x != 0], axis=1)

            d['row'].extend(np.repeat(range(curr_start, next_start), df.col.str.len()).tolist())
            d['col'].extend(np.concatenate(df.col.values).tolist())
            d['data'].extend(np.concatenate(df.data.values).tolist())
            curr_start = next_start
            
        del df
        self.rows = np.frombuffer(d['row'], dtype=np.int32)
        self.cols = np.frombuffer(d['col'], dtype=np.int32)
        self.data = np.frombuffer(d['data'], dtype=np.float32)
        self.ratings_file = os.path.join(self.all_datasets_dir, self.dataset_dir, self.data_file)
        with open(self.ratings_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            writer.writerows(zip(*d.values()))

    def read_interactions(self, file, use_cols, delimiter, header, verbose):
        """Skip reading through the ratings file again"""
        try:
            return self.rows, self.cols, self.data
        except AttributeError:
            return super(Jester, self).read_interactions(file, use_cols, delimiter, header, verbose)


if __name__ == '__main__':
    reader = Jester(use_local=True, force_rebuild=True, implicit=True, save_local=True, verbose=True,
                     min_ratings=1, remove_top_pop=0.0)
    URM = reader.get_URM_full()
    print(URM.nnz)
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