#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import itertools
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datasets.LastFM import LastFM
from datasets.CiaoDVD import CiaoDVD
from datasets.Delicious import Delicious
from datasets.Movielens import Movielens
from datasets.DataReader import DataReader

sns.set_context('paper', font_scale=1.75)
plt.style.use('fivethirtyeight')

def plot_lorenz_curve(datasets, labels, path=None):
    """
    Plots the inverse Lorenz curve of every dataset in datasets.

    Parameters:
    -----------

    datasets: list of scipy.sparse.csr_matrix
        List of full URMs, each with dimensions num_items x num_users

    labels: list
        List of labels of each dataset to use as legend
    
    path: str, default None
        Path where to save the plot. If None, will show the plot.
    """

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.xlim = [0, 100]
    ax.ylim = [0, 100]
    ax.set_xlabel('Percentage of items', fontsize=20)
    ax.set_ylabel('Cumulative percentage of ratings', fontsize=20)

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])

    for i, d in enumerate(datasets):
        # Compute the number of interactions for each item
        item_ratings = np.ediff1d(d.indptr)

        # Sort item interactions from highest to lowest
        sorted_ratings = np.sort(item_ratings)[::-1]

        # Cummulative sum of normalized interactions so we can plot them as probability
        cumulative_ratings = np.cumsum(sorted_ratings) / np.sum(sorted_ratings)

        # Cummulative sum of items so we can plot them as probability
        cumulative_items = np.cumsum(np.ones(d.shape[0])) / d.shape[0]

        # Add Lorenz curve plot
        ax.plot(cumulative_items, cumulative_ratings, label=labels[i], linewidth=2, marker=next(marker), markevery=len(cumulative_items)//20, markersize=8)

    # Add the 45% line
    ax.plot(cumulative_items[0:-1:10], cumulative_items[0:-1:10], label='Uniform distribution', linewidth=2, linestyle='dashed')

    # Add the vertical line at 33% for short-head items
    ax.plot(np.ones(120) * .33, np.linspace(-0.01, 1.01, num=120), label='short-head threshold', color='black', linewidth=2, linestyle='dashdot')

    ax.legend(loc='center right', fontsize='x-large')

    if path is None:
        plt.show()
    else:
        fig.savefig(os.path.join(path, 'lorenz_curve.png'), bbox_inches='tight')

def plot_long_tail(dataset, label, path=None):
    """
    Plots the long tail distribution of a dataset.

    Parameters:
    -----------

    dataset: scipy.sparse.csr_matrix
        URM with dimensions num_items x num_users

    path: str, default None
        Path where to save the plot. If None, will show the plot.
    """

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.xlim = [0, 100]
    ax.set_xlabel('Percentage of items', fontsize=20)
    ax.set_ylabel('Ratings', fontsize=20)

    # Cummulative sum of items so we can plot them as probability
    cumulative_items = np.cumsum(np.ones(dataset.shape[0])) / dataset.shape[0]

    # Compute the number of interactions for each item
    item_ratings = np.ediff1d(dataset.indptr)

    # Sort item interactions from highest to lowest
    sorted_ratings = np.sort(item_ratings)[::-1]

    # PLot long tail and fill
    ax.fill_between(cumulative_items, sorted_ratings, where=cumulative_items<=.33, alpha=.75, color='red')
    ax.fill_between(cumulative_items, sorted_ratings, where=cumulative_items>.33, alpha=.75, color=('#f8de73'))

    # Add horizontal line of short-head items at 33%
    ax.plot(np.ones(100) * .33, np.linspace(-sorted_ratings[0] * .01, sorted_ratings[0] * 1.01, num=100),
            label='short-head threshold', color='black', linewidth=2, linestyle='dashdot')
    
    ax.legend(loc='upper right', fontsize='x-large')

    if path is None:
        plt.show()
    else:
        fig.savefig(os.path.join(path, 'long_tail.png'), bbox_inches='tight')

if __name__ == "__main__":
    # Generic parameters for each dataset
    kwargs = {}
    kwargs['use_local'] = True
    kwargs['force_rebuild'] = False
    kwargs['implicit'] = False
    kwargs['save_local'] = True
    kwargs['verbose'] = False
    kwargs['split'] = False

    datasets = [CiaoDVD, Delicious, LastFM, '100K', '1M', '10M']
    URMs = []

    path = os.path.join(DataReader.all_datasets_dir, 'stats')
    if not os.path.exists(path):
        os.makedirs(path)

    for d in datasets:
        if isinstance(d, str):
            reader = Movielens(version=d, **kwargs)
        else:
            reader = d(**kwargs)
        reader.describe(save_plots=True, path=path)
        URMs.append(reader.get_URM_full().T.tocsr())

    reader = Movielens('1M', **kwargs)
    plot_long_tail(dataset=reader.get_URM_full().T.tocsr(), label=reader.DATASET_NAME, path=path)
    plot_lorenz_curve(datasets=URMs, labels=['MovieLens ' + d if isinstance(d, str) else d.DATASET_NAME for d in datasets], path=path)