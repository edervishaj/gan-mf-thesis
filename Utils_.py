#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import random
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

from sklearn.metrics.pairwise import cosine_similarity


CONSTANTS = dict(root_dir=os.path.dirname(os.path.abspath(__file__)))


def cos_sim(list_vec1, list_vec2):
    """ Element-wise cosine similarity between two lists of vectors """
    sim = np.array([])
    for vec1, vec2 in zip(list_vec1, list_vec2):
        sim = np.append(sim, cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)).flatten())
    return np.mean(sim)


def plot_loss_acc(model, dict_values, xlabel='epochs', ylabel=None, scale='linear'):
    """
    Plots training loss and accuracy values for Discriminator and Generator.

    Parameters
    ----------
    model:
        Recommendation model used (must be GAN-based).

    dict_values: dict
        Dictionary where each key is to be used in the legend.

    xlabel: str, default `epochs`
        Label to use for the x-axis.
    
    ylabel: str, default None
        Label to use for the y-axis.

    scale: str, default `linear`
        Scale to use for plotting. Options are `linear` and `log`.
    """

    if scale != 'log':
        scale = 'linear'

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])
    keys = list(dict_values.keys())
    epochs = len(dict_values[keys[0]])
    fig = plt.figure(figsize=(20, 10))
    plt.xlabel(xlabel)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel)
    plt.grid(True)

    for k in keys:
        if scale == 'log':
            plt.plot(range(epochs), np.log(dict_values[k]), label=k, linestyle='-', alpha=0.8, marker=next(marker))
        else:
            plt.plot(range(epochs), dict_values[k], label=k, linestyle='-', alpha=0.8, marker=next(marker))

    plt.legend(keys, loc='upper right')

    title = 'Loss function of model ' + model.RECOMMENDER_NAME + '\n'
    title += '{'

    config_list = ['d_nodes', 'g_nodes', 'g1_nodes', 'g2_nodes', 'd_hidden_act', 'g_hidden_act', 'g_output_act',
                   'use_dropout', 'use_batchnorm', 'dropout', 'batch_mom', 'epochs', 'sgd_var', 'adam_var', 'sgd_mom', 'beta1']

    for c in model.config.keys():
        if c in config_list:
            title += c + ':' + str(model.config[c]) + ', '

    title = title[:-2]
    title += '}'

    plt.title(title)
    
    save_path = os.path.join(model.logsdir, 'loss' + '_epochs_' + str(epochs) + '_' + str(model.seed) + '.png')
    fig.savefig(save_path, bbox_inches="tight")


def plot_generator_ratings(ratings, rec, neg=False):
    '''
    Plots the mean and std of the fake ratings of batch as received by the generator
    during training.

    :param ratings: List of fake ratings in form [[batch_size, 1], [batch_size, 1], ...]
    :param rec: GAN Model that generated the ratings
    '''

    data = pd.DataFrame(columns=['epoch', 'rating'])
    for e, r in enumerate(ratings):
        epoch_data = (np.ones(r.shape[0], dtype=np.int32) * e).tolist()
        rating_data = r.flatten().tolist()
        tmp_df = pd.DataFrame([[x[0], x[1]] for x in zip(epoch_data, rating_data)], columns=['epoch', 'rating'])
        data = data.append(tmp_df, ignore_index=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.relplot(x='epoch', y='rating', data=data, ci='sd', kind='line', ax=ax)

    if neg:
        save_path = os.path.join(rec.logsdir, 'fake_ratings_neg' + str(rec.seed) + '.png')
    else:
        save_path = os.path.join(rec.logsdir, 'fake_ratings_' + str(rec.seed) + '.png')
    fig.savefig(save_path, bbox_inches="tight")


def plot_gradients(gradients):
    """
    Ridgeplot of gradients over training epochs

    Parameters
    ----------
    gradients: np.ndarray of elements (epoch_number, layer, node_gradient)
        Array of gradients
    """

    pal = sns.cubehelix_palette(n_colors=16, start=0.3, rot=-0.5, light=.7)

    # We have to create a pd.DataFrame in order to use Seaborn.FacetGrid for the ridgeplot.
    epochs = np.unique(gradients[:, 0])
    layers = np.unique(gradients[:, 1])
    fig, ax = plt.subplots(1, len(layers), figsize=(20, 10))
    df = pd.DataFrame(gradients, columns=['epochs', 'layer', 'gradients'])
    for i, l in enumerate(layers):
        g = sns.FacetGrid(df.iloc[:, df.layer == l], row='epochs', hue='epochs', aspect=15, height=.5, palette=pal, ax=ax[0,i])
        g.map(sns.kdeplot, 'gradients', clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)
        g.fig.subplots_adjust(hspace=-.25)
        g.set_titles('')
        g.set(yticks=[])
        g.despine(bottom=True, left=True)


    pass


def plot(feed, title, save_dir, xlabel='epochs', ylabel=None):
    """
    Plots the dictionary provided. Each key is considered a separate line.

    Parameters
    ----------
    feed: dict
        Keys of the dictionary are used in the legend of the plot.

    title: str
        Title of the plot. Also the filename of the plot.

    save_dir: str
        Directory where to save the plot.

    xlabel: str
        Label to be used for the x-axis of the plot.

    ylabel: str, default None
        Label to be used for the y-axis of the plot.
    """

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])
    keys = list(feed.keys())
    fig = plt.figure(figsize=(20, 10))
    plt.xlabel(xlabel)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel)
    plt.grid(True)

    for k in keys:
        data = feed[k]
        plt.plot(range(1, len(data)+1), data, label=k, linestyle='-', alpha=0.8, marker=next(marker))

    plt.legend(keys, loc='upper left')

    plt.title(title)

    save_path = os.path.join(save_dir, title + '.png')
    fig.savefig(save_path, bbox_inches="tight")
