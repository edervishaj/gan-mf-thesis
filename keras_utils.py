#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import json
import random
import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.layers import Layer
from keras.callbacks import TensorBoard
from keras.models import Model, load_model, model_from_json
from keras.regularizers import Regularizer
from keras.utils import CustomObjectScope, plot_model


class Conv1dInv(Layer):
    """Keras Layer defining a 1D column kernel
    multiplied in a column-vector fashion to upsample the input.

    # Properties:
        units: number of output nodes
    """
    def __init__(self, units, **kwargs):
        self.units = units
        super(Conv1dInv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                    shape=(1, self.units),
                                    initializer='glorot_uniform',
                                    trainable=True)
        super(Conv1dInv, self).build(input_shape)

    def call(self, x):
        block_vector = K.permute_dimensions(K.repeat(self.kernel, K.shape(x)[0]), [1, 0, 2])    # (x.shape[0], 1, self.units)
        upsampling = K.batch_dot(K.expand_dims(x, axis=-1), block_vector)
        new_shape = K.stack([K.shape(x)[0], K.shape(x)[1] * self.units, 1])
        return K.reshape(K.expand_dims(upsampling), new_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*self.units, 1)

    def get_config(self):
        base_config = super(Conv1dInv, self).get_config()
        base_config['units'] = self.units
        return base_config


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


class KerasVariableScheduler(object):
    """Performs changing of a Keras variable with a given operation in epochs."""

    def __init__(self, factor, freq, op='mult'):
        """Constructor

        Parameters
        ----------
        factor: float
            Factor by which to change the variable.

        freq: int
            Frequency in epochs that decaying should be applied.

        op: str, default `mult`
            Operation to apply to the variable. Options are `mult` and `add`.
        """
        self.factor = factor
        self.freq = freq
        self.op = op

    def decay(self, epoch, var):
        if epoch % self.freq == 0:
            if self.op == 'add':
                K.set_value(var, K.get_value(var) + self.factor)
            else:
                K.set_value(var, K.get_value(var) * self.factor)

    def __call__(self, epoch, var):
        self.decay(epoch, var)


class EarlyStoppingScheduler(object):
    """Performs early stopping mechanism according to a fixed number of worse evaluations on a validation set."""
    def __init__(self, model, evaluator, metrics=['PRECISION', 'RECALL', 'MAP', 'NDCG'], freq=1, allow_worse=5, custom_objects={}):
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
        """

        self.model = model
        self.evaluator = evaluator
        self.metrics = metrics
        self.freq = freq
        self.best_scores = np.zeros(len(metrics))
        self.allow_worse = allow_worse
        self.worse_left = allow_worse
        self.custom_objects = custom_objects
        self.scores = []

    def score(self, epoch):
        if epoch % self.freq == 0:
            results_dic, _ = self.evaluator.evaluateRecommender(self.model)
            curr_scores = np.array([results_dic[5][m] for m in self.metrics])
            self.scores.append(curr_scores)
            if np.all(np.less_equal(curr_scores, self.best_scores)):
                if self.worse_left > 0:
                    self.worse_left -= 1
                else:
                    self.model.stop_fit()
                    self.load_best()
            else:
                self.best_scores = curr_scores
                self.worse_left = self.allow_worse
                save_models(self.model, save_dir='best_early_stopping')

    def __call__(self, epoch):
        self.score(epoch)

    def load_best(self):
        load_models(self.model, 'best_early_stopping', self.custom_objects)

    def get_scores(self):
        return self.scores


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


def plot_models(rec):
    """
    Plots the Keras models.

    Parameters
    ----------
    rec: BaseRecommender TODO: Implement base recommender
        BaseRecommender model with Keras models instantiated.
    """

    save_path = os.path.join(rec.logsdir, 'archs')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    rec_attributes = dir(rec)
    for attr in rec_attributes:
        keras_model = getattr(rec, attr)
        if isinstance(keras_model, Model):
            plot_model(keras_model, to_file=os.path.join(save_path, keras_model.name+'.png'), show_shapes=True, show_layer_names=True)


def save_models(rec, save_dir='models'):
    """
    Saves the trained Keras models given the Recommendation model.

    :param rec: Recommendation model
    :param save_dir: Directory where to save the models, Default `models`
    """

    save_path = os.path.join(rec.logsdir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    rec_attributes = dir(rec)
    for attr in rec_attributes:
        keras_model = getattr(rec, attr)
        if isinstance(keras_model, Model):
            # keras_model.save(os.path.join(save_path, attr + '.h5'))
            fp = os.path.join(save_path, attr + '.h5')
            model_weights = keras_model.save_weights(fp, overwrite=True)
            model_json = keras_model.to_json()
            with open(os.path.join(save_path, attr + '.json'), 'w') as f:
                f.write(model_json)


def load_models(rec, save_dir='models', custom_objects={}, all_in_folder=False):
    """
    Loads the saved Keras models given the Recommendation model.

    :param rec: Recommendation model
    :param save_dir: Directory from where to retrieve the models. Default `models`
    :param custom_objects: Custom objects (layer, loss func.) required by `keras.model.load_model`
    :param all_in_folder: Flag whether to load all models in the folder or only those already defined in `rec`
    """
    save_path = os.path.join(rec.logsdir, save_dir)
    if not os.path.exists(save_path):
        raise OSError('Folder not found!')

    if all_in_folder:
        for f in os.listdir(save_path):
            if f.endswith('.h5'):
                attr = os.path.basename(f)

                # Load model first from json
                with open(os.path.join(save_path, attr + '.json'), 'r') as fp:
                    setattr(rec, attr, model_from_json(json.dumps(json.load(fp)), custom_objects))

                # Load model weights
                getattr(rec, attr).load_weights(os.path.join(save_path, f))

    else:
        rec_attributes = dir(rec)
        for attr in rec_attributes:
            keras_model = getattr(rec, attr)
            if isinstance(keras_model, Model):
                # Load model first from json
                with open(os.path.join(save_path, attr + '.json'), 'r') as fp:
                    setattr(rec, attr, model_from_json(json.dumps(json.load(fp)), custom_objects))

                # Load model weights
                getattr(rec, attr).load_weights(os.path.join(save_path, attr + '.h5'))


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


def freeze_model(model, unfreeze=False):
    """Freezes keras.Model layers"""
    if unfreeze:
        for l in model.layers:
            l.trainable = True
    else:
        for l in model.layers:
            l.trainable = False