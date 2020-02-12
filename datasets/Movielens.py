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

class Movielens(DataReader):
    """
    This class reads/downloads any of the version of Movielens Datasets
    and provides methods to access 3 sparse matrices URM_train, URM_validation and URM_test
    or create cross-validation folds from the full URM.
    """
    
    DATASET_NAME = 'Movielens'


    urls = {
        '100K':     'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
        '1M':       'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        '10M':      'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
        '20M':      'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
        'small':    'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'latest':   'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
    }

    data_files = {
        '100K':     'ml-100k/u.data',
        '1M':       'ml-1m/ratings.dat',
        '10M':      'ml-10M100K/ratings.dat',
        '20M':      'ml-20m/ratings.csv',
        'small':    'ml-latest-small/ratings.csv',
        'latest':   'ml-latest/ratings.csv'
    }

    separators = {
        '100K':     '\t',
        '1M':       '::',
        '10M':      '::',
        '20M':      ',',
        'small':    ',',
        'latest':   ','
    }


    def __init__(self, version='10M', split=True, **kwargs):
        """
        Constructor

        Parameters
        ----------
        version: str, default `10M`
            The Movielens dataset version to use. Accepted options are `100K`, `1M`, `10M`, `20M`, `small` and `latest`.

        kwargs: dict
            Keyword arguments that go into the constructor of the superclass constructor
        """

        super(Movielens, self).__init__(delim=self.separators[version], **kwargs)

        if version not in self.urls.keys():
            raise KeyError(version + ' is not supported. Accepted Movielens versions are ' + ', '.join(list(self.urls.keys())) + '.')

        self.version = version
        self.DATASET_NAME = 'Movielens' + self.version
        tmp = self.data_files[self.version].split(os.path.sep)
        self.dataset_dir = tmp[0]
        self.data_file = tmp[1]

        try:
            self.config['version'] = self.version
        except AttributeError:
            pass

        self.process(split)


    def get_ratings_file(self):
        """
        Downloads the Movielens version specified by self.version
        """

        try:
            url = self.urls[self.version]
            zip_file = self.download_url(url, self.verbose, desc='Downloading Movielens from ')
            zfile = zipfile.ZipFile(zip_file)
            to_extract = self.data_files[self.version]
            self.ratings_file = zfile.extract(to_extract, self.all_datasets_dir)
            os.remove(zip_file)
        except (FileNotFoundError, zipfile.BadZipFile):
            print('Either file ' + to_extract + ' not found or ' + os.path.split(self.urls[self.version])[-1] + ' is corrupted', file=sys.stderr)
            raise