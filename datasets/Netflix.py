#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import pickle
import pandas as pd
import scipy.sparse as sps
from datasets.DataReader import DataReader

url = 'https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz'


class Netflix(DataReader):
    """
    This class uses kaggle python package to download the Netflix Prize dataset
    and construct 3 sparse matrices URM_train, URM_validation and URM_test
    """

    def __init__(self,
                 dataset='netflix-inc/netflix-prize-data',
                 use_cols={'user_id': 0, 'item_id': 1, 'rating': 2},
                 split_ratio=[0.6, 0.2, 0.2],
                 delimiter=',',
                 header=False,
                 implicit=False,
                 use_local=True,
                 force_rebuild=False,
                 save_local=True,
                 verbose=True):

        super(Netflix, self).__init__(use_cols, split_ratio, header, implicit, use_local, force_rebuild, save_local, verbose)

        self.dataset = dataset
        self.dataset_dir = os.path.join(self.all_datasets_dir, self.dataset.split('/')[-1])
        self.delim = delimiter
        self.config['delim'] = delimiter

        # Check if files URM_train, URM_test and URM_validation already exists first
        # If not, build locally the sparse matrices using the ratings' file
        if self.use_local:
            train_path = os.path.join(self.dataset_dir, 'URM_train.npz')
            test_path = os.path.join(self.dataset_dir, 'URM_test.npz')
            valid_path = os.path.join(self.dataset_dir, 'URM_validation.npz')

            # Read the build config and compare with current build
            config_path = os.path.join(self.dataset_dir, 'config.pkl')
            if os.path.isfile(config_path):
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)

                if self.config != config:
                    if self.verbose:
                        print('Local matrices built differently from requested build. Setting force_rebuild = True.')
                    self.force_rebuild = True

            else:
                if self.verbose:
                    print('Configuration file not found. Setting force_rebuild = True.')
                self.force_rebuild = True

            if os.path.isfile(train_path) and os.path.isfile(test_path) and os.path.isfile(valid_path) and not self.force_rebuild:
                if self.verbose:
                    print('Loading train, test and validation matrices locally...')

                self.URM_train = sps.load_npz(train_path)
                self.URM_test = sps.load_npz(test_path)
                self.URM_validation = sps.load_npz(valid_path)

            else:
                if self.verbose:
                    print("Matrices not found or rebuilding asked. Building from ratings' file...")
                self.build_local()

                # Save build config locally
                with open(os.path.join(self.dataset_dir, 'config.pkl'), 'wb') as f:
                    pickle.dump(self.config, f)

        else:
            self.build_remote()


    def build_remote(self):
        self.download_kaggle_dataset(dataset=self.dataset, verbose=self.verbose)
        self.ratings_file = os.path.join(self.dataset_dir, 'combined.txt')

        if os.path.isfile(self.ratings_file):
            self.URM_train, \
            self.URM_test, \
            self.URM_validation = self.sps_matrices_from_file(fname=self.ratings_file,
                                                              use_cols=self.use_cols,
                                                              delimiter=self.delim,
                                                              header=self.header,
                                                              split_ratio=self.split_ratio,
                                                              save_local=self.save_local,
                                                              implicit=self.implicit,
                                                              verbose=self.verbose,
                                                              netflix_process=True)


    def build_local(self):
        self.ratings_file = os.path.join(self.dataset_dir, 'combined_data_1.txt')

        if os.path.isfile(self.ratings_file):
            self.URM_train,\
            self.URM_test,\
            self.URM_validation = self.sps_matrices_from_file(fname=self.ratings_file,
                                                              use_cols=self.use_cols,
                                                              delimiter=self.delim,
                                                              header=self.header,
                                                              split_ratio=self.split_ratio,
                                                              save_local=self.save_local,
                                                              implicit=self.implicit,
                                                              verbose=self.verbose,
                                                              netflix_process=True)
        else:
            print(self.ratings_file + ' not found. Building remotely...')
            self.build_remote()



if __name__ == '__main__':
    net = Netflix(use_local=True)
    URM_train = net.get_URM_train()
    URM_test = net.get_URM_test()
    URM_valid = net.get_URM_validation()
    print(URM_train.shape)
    print(URM_test.shape)
    print(URM_valid.shape)