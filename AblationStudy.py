#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
plt.style.use('fivethirtyeight')
sns.set_context('paper', font_scale=1.75)

from GANRec.GANMF import GANMF
from Base.Evaluation.Evaluator import EvaluatorHoldout
from RecSysExp import load_URMs, set_seed, dataset_kwargs, all_datasets, name_datasets
from RunBestParameters import load_best_params

seed = 1337
metrics = ['PRECISION', 'RECALL', 'NDCG', 'MAP']

def ablation_study(arguments):
    study_path = 'ablation_study'
    if not os.path.exists(study_path):
        os.makedirs(study_path, exist_ok=False)

    exp_path = 'experiments'
    datasets = []
    modes = ['user', 'item']
    run_all = False

    if '--run-all' in arguments:
        datasets = all_datasets
        run_all = True

    for arg in arguments:
        if arg in name_datasets and not run_all:
            datasets.append(all_datasets[name_datasets.index(arg)])
        if arg in modes:
            modes = [arg]

    cutoffs = [5, 10, 20, 50]

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])

    for m in modes:
        for d in datasets:
            plotting_data = {c: {m: [] for m in metrics} for c in cutoffs}
            best_params = load_best_params(exp_path, d if isinstance(d, str) else d.DATASET_NAME, 'GANMF', m)
            range_coeff = np.arange(0, 1.1, 0.2)
            for coeff in range_coeff:
                best_params['recon_coefficient'] = coeff
                URM_train, URM_test, _, _, _ = load_URMs(d, dataset_kwargs)
                set_seed(seed)
                test_evaluator = EvaluatorHoldout(URM_test, cutoffs, exclude_seen=True)
                model = GANMF(URM_train, mode=m, seed=seed, is_experiment=True)
                model.fit(validation_set=None, sample_every=None, validation_evaluator=None, **best_params)
                result_dict, result_str = test_evaluator.evaluateRecommender(model)
                plotting_data[coeff] = {}
                for c in cutoffs:
                    for met in metrics:
                        plotting_data[c][met].append(result_dict[c][met])

            dname = d if isinstance(d, str) else d.DATASET_NAME
            substudy_path = os.path.join(study_path, dname + '_GANMF_' + m)
            if not os.path.exists(substudy_path):
                os.makedirs(substudy_path, exist_ok=False)

            for c in cutoffs:
                fig, ax = plt.subplots(figsize=(20, 10))
                ax.set_xlabel('Feature Matching Coefficient')
                for met in metrics:
                    ax.plot(range_coeff, plotting_data[c][met], label=met, marker=next(marker))
                ax.legend(loc='best', fontsize='x-large')
                fig.savefig(os.path.join(substudy_path, str(c) + '_feature_matching_effect.png'), bbox_inches='tight')

def feature_matching_cos_sim(arguments):
    path_cos_sim = 'cosine_similarities'
    if not os.path.exists(path_cos_sim):
        os.makedirs(path_cos_sim, exist_ok=False)

    exp_path = 'experiments'
    datasets = []

    run_all = False

    if '--run-all' in arguments:
        datasets = all_datasets
        run_all = True

    for arg in arguments:
        if arg in name_datasets and not run_all:
            datasets.append(all_datasets[name_datasets.index(arg)])

    for d in datasets:
        dname =d if isinstance(d, str) else d.DATASET_NAME
        best_params = load_best_params(exp_path, dname, 'GANMF', 'user')
        URM_train, _, _, _, _ = load_URMs(d, dataset_kwargs)
        set_seed(seed)
        model = GANMF(URM_train, seed=seed, is_experiment=True)
        model.fit(validation_set=None, sample_every=None, validation_evaluator=None, **best_params)
        all_predictions = model._compute_item_score(user_id_array=np.array(range(URM_train.shape[0])))

        fig, ax = plt.subplots(figsize=(20, 10))
        similarity = cosine_similarity(all_predictions)
        mean = np.mean(similarity)
        std = np.std(similarity)
        sns.heatmap(similarity, vmin=-1, vmax=1, ax=ax)
        ax.tick_params(left=False, bottom=False)
        hm_save_path = os.path.join(path_cos_sim,
                                    'GANMF' + '_' + dname + '_with_fm.png')
        stats_save_path = os.path.join(path_cos_sim,
                                       'GANMF' + '_' + dname + '_with_fm.txt')
        fig.savefig(hm_save_path, bbox_inches="tight")
        with open(stats_save_path, 'a') as f:
            f.write('Mean: ' + str(mean))
            f.write('\n')
            f.write('Std: ' + str(std))

        best_params['recon_coefficient'] = 0
        set_seed(seed)
        model = GANMF(URM_train, seed=seed, is_experiment=True)
        model.fit(validation_set=None, sample_every=None, validation_evaluator=None, **best_params)
        all_predictions = model._compute_item_score(user_id_array=np.array(range(URM_train.shape[0])))
        fig, ax = plt.subplots(figsize=(20, 10))
        similarity = cosine_similarity(all_predictions)
        mean = np.mean(similarity)
        std = np.std(similarity)
        sns.heatmap(similarity, vmin=-1, vmax=1, ax=ax)
        ax.tick_params(left=False, bottom=False)
        hm_save_path = os.path.join(path_cos_sim,
                                    'GANMF' + '_' + dname + '_wo_fm.png')
        stats_save_path = os.path.join(path_cos_sim,
                                       'GANMF' + '_' + dname + '_wo_fm.txt')
        fig.savefig(hm_save_path, bbox_inches="tight")
        with open(stats_save_path, 'a') as f:
            f.write('Mean: ' + str(mean))
            f.write('\n')
            f.write('Std: ' + str(std))

if __name__ == '__main__':
    # Run this script as `python AblationStudy.py [--run-all] dataset_name [GANMF_mode]`
    assert len(sys.argv) >= 2, 'Number of arguments must be greater than 2, given {:d}'.format(len(sys.argv))
    arguments = sys.argv[1:]
    # ablation_study(arguments)
    feature_matching_cos_sim(arguments)
