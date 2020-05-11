#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import pickle

from GANRec.GANMF import GANMF
from GANRec.CFGAN import CFGAN
from GANRec.DisGANMF import DisGANMF
from GANRec.fullGANMF import fullGANMF
from GANRec.DeepGANMF import DeepGANMF

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Base.NonPersonalizedRecommender import TopPop, Random
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from MatrixFactorization.NMFRecommender import NMFRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython

from RecSysExp import all_datasets, set_seed, load_URMs, dataset_kwargs, all_recommenders, name_datasets

seed = 1337

def load_best_params(path, dataset, recommender, train_mode=''):
    params_path = os.path.join(path, recommender + '_' + train_mode + '_' + dataset, 'best_params.pkl')
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def main(arguments):
    test_results_path = 'test_results'
    if not os.path.exists(test_results_path):
        os.makedirs(test_results_path, exist_ok=False)

    exp_path = 'experiments'
    datasets = []
    run_all = False
    train_mode = ['user', 'item']
    cutoffs = [5, 10, 20, 50]
    recommender = None

    dict_rec_classes = {}
    dict_rec_classes['TopPop'] = TopPop
    dict_rec_classes['Random'] = Random
    dict_rec_classes['PureSVD'] = PureSVDRecommender
    dict_rec_classes['BPR'] = MatrixFactorization_BPR_Cython
    dict_rec_classes['ALS'] = IALSRecommender
    dict_rec_classes['NMF'] = NMFRecommender
    dict_rec_classes['GANMF'] = GANMF
    dict_rec_classes['CFGAN'] = CFGAN
    dict_rec_classes['DisGANMF'] = DisGANMF
    dict_rec_classes['SLIMBPR'] = SLIM_BPR_Cython
    dict_rec_classes['fullGANMF'] = fullGANMF
    dict_rec_classes['DeepGANMF'] = DeepGANMF

    if '--run-all' in arguments:
        datasets = all_datasets
        run_all = True

    for arg in arguments:
        if arg in name_datasets and not run_all:
            datasets.append(all_datasets[name_datasets.index(arg)])
        if arg in ['user', 'item']:
            train_mode = [arg]
        if arg in all_recommenders and recommender is None:
            recommender = arg

    if recommender not in ['GANMF', 'DisGANMF', 'CFGAN', 'fullGANMF', 'DeepGANMF']:
        train_mode = ['']

    for d in datasets:
        dname = d if isinstance(d, str) else d.DATASET_NAME
        for mode in train_mode:
            if recommender == 'fullGANMF':
                best_params = load_best_params(exp_path, dname, 'GANMF', mode)
            else:
                best_params = load_best_params(exp_path, dname, dict_rec_classes[recommender].RECOMMENDER_NAME, mode)
            set_seed(seed)
            URM_train, URM_test, _, _, _ = load_URMs(d, dataset_kwargs)
            test_evaluator = EvaluatorHoldout(URM_test, cutoffs, exclude_seen=True)
            if recommender in ['GANMF', 'DisGANMF', 'CFGAN', 'fullGANMF', 'DeepGANMF']:
                model = dict_rec_classes[recommender](URM_train, mode=mode, seed=seed, is_experiment=True)
                model.fit(validation_set=None, sample_every=None, validation_evaluator=None, **best_params)
            else:
                model = dict_rec_classes[recommender](URM_train)
                model.fit(**best_params)
            results_dict, results_str = test_evaluator.evaluateRecommender(model)

            save_path = os.path.join(test_results_path, model.RECOMMENDER_NAME + '_' + mode + '_' + dname)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=False)
                with open(os.path.join(save_path, 'test_results.txt'), 'a') as f:
                    f.write(results_str)
            else:
                results_filename = os.path.join(save_path, 'test_results.txt')
                if not os.path.exists(results_filename):
                    with open(results_filename, 'a') as f:
                        f.write(results_str)


if __name__ == '__main__':
    # Run this script as `python RunBestParameters.py recommender_name [train_mode] [--run-all] dataset_name `
    assert len(sys.argv) >= 2, 'Number of arguments must be greater than 2, given {:d}'.format(len(sys.argv))
    arguments = sys.argv[1:]
    main(arguments)