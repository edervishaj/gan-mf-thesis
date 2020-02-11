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
import tensorflow as tf
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

import keras.backend as K
from keras.utils import CustomObjectScope
from keras.models import Model, load_model
from keras.layers import Layer, Input, Dense, Embedding, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard
from keras.regularizers import Regularizer

from sklearn.metrics.pairwise import cosine_similarity


CONSTANTS = dict(root_dir=os.path.dirname(os.path.abspath(__file__)))


class OneHotLayer(Layer):
    """Keras Layer that converts integers to one-hot encoded values

    # Properties
        depth: The total number of possible integer values.
    """

    def __init__(self, depth, **kwargs):
        self.depth = depth
        super(OneHotLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return K.one_hot(K.cast(inputs, 'int32'), self.depth)[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.depth)

    def get_config(self):
        base_config = super(OneHotLayer, self).get_config()
        base_config['depth'] = self.depth
        return base_config


class MinibatchDiscriminationLayer(Layer):
    def __init__(self, **kwargs):
        # self.T_shape = T_shape
        super(MinibatchDiscriminationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.T = K.random_uniform(shape=self.T_shape)  # (no_samples, dim, dim)
        # self.T = K.placeholder(shape=self.T_shape)
        super(MinibatchDiscriminationLayer, self).build(input_shape)

    def paper_imp(self, inputs):
        # This is the implementation of "Improved training for GANs"
        # K.set_value(self.T, K.tile(profiles, self.T.shape[0]))
        Ms = K.dot(inputs, K.permute_dimensions(self.T, (1, 0, 2)))  # (N, dim, dim)
        diff = tf.math.subtract(Ms, K.expand_dims(Ms, 1))  # (N, dim, dim, dim)
        CBs = K.exp(-tf.norm(diff, axis=3, ord=1))  # (N, N, dim)
        closeness = K.sum(CBs, axis=1)  # (N, dim)
        return K.concatenate([inputs, closeness], axis=1)  # (N, input_dim + dim)

    def call(self, inputs):
        # Compute the cosine similarity between the batch profiles
        # inputs are N x dim
        # self.cosine_inputs(inputs)
        normalized_inputs = K.l2_normalize(inputs, axis=1) # N x dim
        cosine_sim = K.dot(normalized_inputs, K.permute_dimensions(normalized_inputs, (1, 0))) # N x N
        return K.concatenate([inputs, cosine_sim], axis=1)   # N x dim + N

    def compute_output_shape(self, input_shape):
        # return (input_shape[0], input_shape[1] + self.T_shape[-1])
        return (input_shape[0], input_shape[1] + 32)

    def get_config(self):
        base_config = super(MinibatchDiscriminationLayer, self).get_config()
        # base_config['T_shape'] = self.T_shape
        return base_config


class LearningRateScheduler():
    """Performs decaying of a learning rate with a given frequency in epochs."""

    def __init__(self, factor, freq):
        """Constructor

        Parameters
        ----------
        factor: float
            Factor by which to change the learning rate.

        freq: int
            Frequency in epochs that decaying should be applied.
        """
        self.factor = factor
        self.freq = freq

    def decay(self, epoch, lr):
        if epoch % self.freq == 0:
            K.set_value(lr, K.get_value(lr) * self.factor)

    def __call__(self, epoch, lr):
        self.decay(epoch, lr)


class EarlyStoppingScheduler():
    """Performs early stopping mechanism according to a fixed number of worse evaluations on a validation set."""
    def __init__(self, model, evaluator, metrics=['PRECISION', 'RECALL', 'MAP', 'NDCG'], freq=1, allow_worse=5, custom_objects=None):
        """Constructor

        Parameters
        ----------
        model: BaseRecommender
            Implements _compute_item_score() TODO: change the base class for the models

        evaluator: Evaluator
            Initialized with the validation set.

        metrics: list[str]
            List of metrics present in the evaluator for which early stopping will be evaluated.

        freq: int
            Frequency in epochs when to perform evaluation on validation set.

        allow_worse: int
            Allowed number of bad results on all metrics.

        custom_objects: dict
            Dictionary of custom objects to be used for serialization of the Keras Model(s) during saving of results.
        """

        self.model = model
        self.evaluator = evaluator
        self.metrics = metrics
        self.freq = freq
        self.best_scores = np.zeros(len(metrics))
        self.allow_worse = allow_worse
        self.worse_left = allow_worse
        self.custom_objects = custom_objects

    def score(self, epoch):
        if epoch % self.freq == 0:
            results_dic, _ = self.evaluator.evaluateRecommender(self.model)
            curr_scores = np.array([results_dic[5][m] for m in self.metrics])
            if np.all(np.less_equal(curr_scores, self.best_scores)):
                if self.worse_left > 0:
                    self.worse_left -= 1
                else:
                    print('Training stopped, epoch:', epoch)
                    self.model.stop_fit()
                    self.load_best()
            else:
                self.best_scores = curr_scores
                self.worse_left = self.allow_worse
                save_model(self.model, save_dir='best_early_stopping')

    def __call__(self, epoch):
        self.score(epoch)

    def load_best(self):
        if self.custom_objects is not None:
            load_models(self.model, 'best_early_stopping', custom_objects=self.custom_objects)
        else:
            load_models(self.model, 'best_early_stopping')


class CustomTensorboardCallback(TensorBoard):
    def on_epoch_end(self, epoch, logs=None, inputs=None):
        if inputs is None:
            return
        if epoch % self.histogram_freq == 0:
            tensors = (self.model.inputs + self.model.targets + self.model.sample_weights)
            if self.model.uses_learning_phase:
                tensors += [K.learning_phase()]
            feed_dict = dict(zip(tensors, inputs))
            result = self.sess.run([self.merged], feed_dict=feed_dict)
            summary_str = result[0]
            self.writer.add_summary(summary_str, epoch)


class DynamicL1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.
    Regularization coefficients can be changed dynamically during training.
    # Properties
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.variable(K.cast_to_floatx(l1))
        self.l2 = K.variable(K.cast_to_floatx(l2))

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += self.l1 * K.sum(K.abs(x))
        if self.l2:
            regularization += self.l2 * K.sum(K.square(x))
        return regularization

    def get_config(self):
        return {'l1': K.cast_to_floatx(K.get_value(self.l1)),
                'l2': K.cast_to_floatx(K.get_value(self.l2))}


def l1(l=0.01):
    return DynamicL1L2(l1=l)


def l2(l=0.01):
    return DynamicL1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return DynamicL1L2(l1=l1, l2=l2)


# def ZeroReconstructionLoss(no_indices):
#     def loss(y_true, y_pred):
#         where_is_zero = tf.where(K.equal(y_true, 0))[0]
#         selected_indices = K.cast(K.random_binomial(shape=K.shape(where_is_zero), p=percent, dtype='float32'), 'int32')
#         selected_zeros = K.gather(where_is_zero, selected_indices)
#         return K.mean(K.square(0 - K.gather(y_pred, selected_zeros)))
#     return loss


def ZeroReconstructionLoss(mask):
    """Keras custom loss for CFGAN zero reconstruction of generated purchase vectors.

    Parameters
    ----------
    mask: tf.Tensor with shape batch_size x len(purchase_vector)
        Binary tensor where ones represent users/items for which to compute the zero reconstruction loss and
        zeros otherwise.
    """
    def loss(y_true, y_pred):
        return K.sum(K.square(K.zeros_like(y_pred) - y_pred) * mask, axis=1)
    return loss


def generateGAN(latent_dim=100, data_dim=100, embedding_dim=0, g_nodes=[32], d_nodes=[32], g_hidden_act='LeakyReLU',
                d_hidden_act='LeakyReLU', g_output_act='tanh', d_output_act='sigmoid', verbose=False):
    # Create the discriminator
    d_input = Input(shape=(data_dim,), name='d_input')
    d = d_input
    for i in range(len(d_nodes) - 1):
        nodes = d_nodes[i]
        d = Dense(units=nodes, activation=d_hidden_act)(d)
        if d_hidden_act == 'LeakyReLU':
            d = LeakyReLU(0.2)(d)
        else:
            d = Activation(d_hidden_act)(d)
    d = Dense(units=d_nodes[-1], activation=d_output_act)(d)
    discriminator = Model(d_input, d, name='discriminator')

    # Create generators
    if embedding_dim > 0:
        g_input = Embedding(input_dim=latent_dim + 1, output_dim=embedding_dim, input_length=1)
    else:
        g_input = Input(shape=(latent_dim,), name='g1_input')
    g = g_input
    for i in range(len(g_nodes) - 1):
        nodes = g_nodes[i]
        g = Dense(units=nodes, activation=g_hidden_act)(g)
        if g_hidden_act == 'LeakyReLU':
            g = LeakyReLU(0.2)(g)
        else:
            g = Activation(g_hidden_act)(g)
    g = Dense(units=g_nodes[-1], activation=g_output_act)
    gen = Model(g_input, g, name='gen')

    if verbose:
        print('Discriminator network:')
        discriminator.summary()
        print('\n\nGenerator network:')
        gen.summary()

    return discriminator, gen


def cos_sim(list_vec1, list_vec2):
    """ Element-wise cosine similarity between two lists of vectors """
    sim = np.array([])
    for vec1, vec2 in zip(list_vec1, list_vec2):
        sim = np.append(sim, cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)).flatten())
    return np.mean(sim)


def set_seed(seed):
    """
    Sets the seed for Python, Numpy, Tensorflow in order to reproduce results.

    Parameters
    ----------
    seed: integer
        Integer value to be used as seed
    """

    # Seed for reproducibility of results
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_model(rec):
    """
    Plots the Keras models give as iterable argument.

    :param rec: Recommendation model
    :return:
    """

    save_path = os.path.join(rec.logsdir, 'archs')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    from keras.utils import plot_model
    rec_attributes = dir(rec)
    if 'discriminator' in rec_attributes:
        plot_model(rec.discriminator, to_file=os.path.join(save_path, rec.discriminator.name+'.png'), show_shapes=True, show_layer_names=True)
    if 'generators' in rec_attributes:
        plot_model(rec.generators, to_file=os.path.join(save_path, rec.generators.name+'.png'), show_shapes=True, show_layer_names=True)
    if 'combined' in rec_attributes:
        plot_model(rec.combined, to_file=os.path.join(save_path, rec.combined.name+'.png'), show_shapes=True, show_layer_names=True)
    if 'combined1' in rec_attributes:
        plot_model(rec.combined1, to_file=os.path.join(save_path, rec.combined1.name+'.png'), show_shapes=True, show_layer_names=True)
    if 'combined2' in rec_attributes:
        plot_model(rec.combined2, to_file=os.path.join(save_path, rec.combined2.name+'.png'), show_shapes=True, show_layer_names=True)
    if 'disc1' in rec_attributes:
        plot_model(rec.disc1, to_file=os.path.join(save_path, rec.disc1.name+'.png'), show_shapes=True, show_layer_names=True)
    if 'disc2' in rec_attributes:
        plot_model(rec.disc2, to_file=os.path.join(save_path, rec.disc2.name+'.png'), show_shapes=True, show_layer_names=True)
    if 'gen1' in rec_attributes:
        plot_model(rec.gen1, to_file=os.path.join(save_path, rec.gen1.name + '.png'), show_shapes=True, show_layer_names=True)
    if 'gen2' in rec_attributes:
        plot_model(rec.gen2, to_file=os.path.join(save_path, rec.gen2.name + '.png'), show_shapes=True, show_layer_names=True)
    if 'trainable_noise' in rec_attributes:
        plot_model(rec.trainable_noise, to_file=os.path.join(save_path, rec.trainable_noise.name + '.png'), show_shapes=True, show_layer_names=True)


def save_model(rec, save_dir='models'):
    """
    Saves the trained Keras models given the Recommendation model.

    :param rec: Recommendation model
    """

    save_path = os.path.join(rec.logsdir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    rec_attributes = dir(rec)
    if 'discriminator' in rec_attributes:
        rec.discriminator.save(os.path.join(save_path, 'discriminator.h5'))
    if 'generators' in rec_attributes:
        rec.generators.save(os.path.join(save_path, 'generators.h5'))
    if 'combined' in rec_attributes:
        rec.combined.save(os.path.join(save_path, 'combined.h5'))
    if 'combined1' in rec_attributes:
        rec.combined1.save(os.path.join(save_path, 'combined1.h5'))
    if 'combined2' in rec_attributes:
        rec.combined2.save(os.path.join(save_path, 'combined2.h5'))
    if 'disc1' in rec_attributes:
        rec.disc1.save(os.path.join(save_path, 'disc1.h5'))
    if 'disc2' in rec_attributes:
        rec.disc2.save(os.path.join(save_path, 'disc2.h5'))
    if 'gen1' in rec_attributes:
        rec.gen1.save(os.path.join(save_path, 'gen1.h5'))
    if 'gen2' in rec_attributes:
        rec.gen2.save(os.path.join(save_path, 'gen2.h5'))
    if 'trainable_noise' in rec_attributes:
        rec.trainable_noise.save(os.path.join(save_path, 'trainable_noise.h5'))


def load_models(rec, save_dir='models', custom_objects=None):
    save_path = os.path.join(rec.logsdir, save_dir)
    if not os.path.exists(save_path):
        raise OSError('Folder not found!')

    rec_attributes = dir(rec)
    if custom_objects is not None:
        with CustomObjectScope(custom_objects):
            if 'discriminator' in rec_attributes:
                rec.discriminator = load_model(os.path.join(save_path, 'discriminator.h5'), compile=False)
            if 'generators' in rec_attributes:
                rec.generators = load_model(os.path.join(save_path, 'generators.h5'), compile=False)
            if 'combined' in rec_attributes:
                rec.combined = load_model(os.path.join(save_path, 'combined.h5'), compile=False)
            if 'combined1' in rec_attributes:
                rec.combined1 = load_model(os.path.join(save_path, 'combined1.h5'), compile=False)
            if 'combined2' in rec_attributes:
                rec.combined2 = load_model(os.path.join(save_path, 'combined2.h5'), compile=False)
            if 'disc1' in rec_attributes:
                rec.disc1 = load_model(os.path.join(save_path, 'disc1.h5'), compile=False)
            if 'disc2' in rec_attributes:
                rec.disc2 = load_model(os.path.join(save_path, 'disc2.h5'), compile=False)
            if 'gen1' in rec_attributes:
                rec.gen1 = load_model(os.path.join(save_path, 'gen1.h5'), compile=False)
            if 'gen2' in rec_attributes:
                rec.gen2 = load_model(os.path.join(save_path, 'gen2.h5'), compile=False)
            if 'trainable_noise' in rec_attributes:
                rec.trainable_noise = load_model(os.path.join(save_path, 'trainable_noise.h5'), compile=False)
    else:
        if 'discriminator' in rec_attributes:
            rec.discriminator = load_model(os.path.join(save_path, 'discriminator.h5'))
        if 'generators' in rec_attributes:
            rec.generators = load_model(os.path.join(save_path, 'generators.h5'))
        if 'combined' in rec_attributes:
            rec.combined = load_model(os.path.join(save_path, 'combined.h5'))
        if 'combined1' in rec_attributes:
            rec.combined1 = load_model(os.path.join(save_path, 'combined1.h5'))
        if 'combined2' in rec_attributes:
            rec.combined2 = load_model(os.path.join(save_path, 'combined2.h5'))
        if 'disc1' in rec_attributes:
            rec.disc1 = load_model(os.path.join(save_path, 'disc1.h5'))
        if 'disc2' in rec_attributes:
            rec.disc2 = load_model(os.path.join(save_path, 'disc2.h5'))
        if 'gen1' in rec_attributes:
            rec.gen1 = load_model(os.path.join(save_path, 'gen1.h5'))
        if 'gen2' in rec_attributes:
            rec.gen2 = load_model(os.path.join(save_path, 'gen2.h5'))
        if 'trainable_noise' in rec_attributes:
            rec.trainable_noise = load_model(os.path.join(save_path, 'trainable_noise.h5'))


def plot_loss_acc(model, dict_values, xlabel='epochs', ylabel=None):
    """
    Plots training loss and accuracy values for Discriminator and Generator.

    Parameters
    ----------
    model:
        Recommendation model used (must be GAN-based)

    dict_values: dict
        Dictionary where each key is to be used in the legend.
    """

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])
    keys = list(dict_values.keys())
    epochs = len(dict_values[keys[0]])
    fig = plt.figure(figsize=(20, 10))
    plt.xlabel(xlabel)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel)
    plt.grid(True)

    for k in keys:
        plt.plot(range(epochs), dict_values[k], label=k, linestyle='-', alpha=0.8, marker=next(marker))

    plt.legend(keys, loc='upper right')

    title = 'Loss function of model ' + model.RECOMMENDER_NAME + '\n'
    title += '{'

    config_list = ['d_nodes', 'g_nodes', 'g1_nodes', 'g2_nodes', 'd_hidden_act', 'g_hidden_act', 'g_output_act',
                   'use_dropout', 'use_batchnorm', 'dropout', 'batch_mom', 'epochs', 'sgd_lr', 'adam_lr', 'sgd_mom', 'beta1']

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


def get_gradients(model):
    """
    Returns the gradients of the model.
    https://github.com/keras-team/keras/issues/2226#issuecomment-259004640

    Also Keras training.py function make_training_function shows how to get gradients.

    Parameters
    ----------
    model: keras.models.Model
        Keras Model for which we have to get the gradients

    Returns
    -------
        List of gradients of the model
    """
    weights = model.trainable_weights
    gradients = model.optimizer.get_gradients(model.total_loss, weights)
    inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    if model._uses_dynamic_learning_phase():
        inputs += [K.learning_phase()]
    return K.function(inputs, gradients)


def tsb_summary_fn(tsb_callback):
    """
    https://groups.google.com/forum/#!topic/keras-users/rEJ1xYqD3AM

    Parameters
    ----------
    tsb_callback: keras.callbacks.TensorBoard
        Keras TensorBoard callback initialized with set_model

    Returns
    -------
    K.function: function that can run the merged summaries given the model inputs
    """
    model = tsb_callback.model
    inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    if model._uses_dynamic_learning_phase():
        inputs += [K.learning_phase()]
    return K.function(inputs, [tsb_callback.merged])


def add_gradients_summary(model):
    """
    Adds keras activations and gradients tensors as histogram summaries.
    (Check keras/callbacks/tensorboard_v1.py on github on how to get activations and gradients)

    Parameters
    ----------
    model: keras.models.Model
        Keras Model for which we have to add gradients summaries
    """

    summaries = []
    for layer in model.layers:
        for weight in layer.trainable_weights:
            mapped_weight_name = weight.name.replace(':', '_')
            summaries.append(tf.summary.histogram(mapped_weight_name, weight))
            if weight in layer.trainable_weights:
                grads = model.optimizer.get_gradients(model.total_loss, weight)

                def is_indexed_slices(grad):
                    return type(grad).__name__ == 'IndexedSlices'

                grads = [grad.values if is_indexed_slices(grad) else grad for grad in grads]
                summaries.append(tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads))
    return tf.summary.merge(summaries)


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
    """

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])
    keys = list(feed.keys())
    x = len(feed[keys[0]])
    fig = plt.figure(figsize=(20, 10))
    plt.xlabel(xlabel)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel)
    plt.grid(True)

    for k in keys:
        plt.plot(range(1, x+1), feed[k], label=k, linestyle='-', alpha=0.8, marker=next(marker))

    plt.legend(keys, loc='upper right')

    plt.title(title)

    save_path = os.path.join(save_dir, title + '.png')
    fig.savefig(save_path, bbox_inches="tight")


def freeze_model(model, unfreeze=False):
    """Freezes keras.Model layers"""
    if unfreeze:
        for l in model.layers:
            l.trainable = True
    else:
        for l in model.layers:
            l.trainable = False