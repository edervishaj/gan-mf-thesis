#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import json
import time
import tqdm
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from Utils_ import EarlyStoppingScheduler
from Base.BaseRecommender import BaseRecommender

class DisGANMF(BaseRecommender):
    RECOMMENDER_NAME = 'DisGANMF'

    def __init__(self, URM_train, mode='user', seed=1234, verbose=False, is_experiment=False):

        if mode not in ['user', 'item']:
            raise ValueError('Accepted training modes are `user` and `item`. Given was {}.', mode)

        self.mode = mode
        if self.mode == 'item':
            self.URM_train = URM_train.T.tocsr()
        else:
            self.URM_train = URM_train
        self.num_users, self.num_items = self.URM_train.shape
        self.config = None
        self.seed = seed
        self.verbose = verbose
        self.logsdir = os.path.join('plots', self.RECOMMENDER_NAME, datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.is_experiment = is_experiment

        if not os.path.exists(self.logsdir) and not self.is_experiment:
            os.makedirs(self.logsdir, exist_ok=False)

        if not self.is_experiment:
            # Save this file inside logsdir/code
            codesdir = os.path.join(self.logsdir, 'code')
            os.makedirs(codesdir, exist_ok=False)
            shutil.copy(os.path.abspath(sys.modules[self.__module__].__file__), codesdir)

    def build(self, d_layers=1, d_nodes=32, d_hidden_act='linear', num_factors=10):
        glorot_uniform = tf.glorot_uniform_initializer()

        ##########################
        # DISCRIMINATOR FUNCTION #
        ##########################
        def discriminator(condition, input_data):
            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                d = tf.concat([condition, input_data], axis=1)
                for l in range(d_layers):
                    d = tf.layers.dense(d, units=d_nodes, kernel_initializer=glorot_uniform, name='layer_' + str(l),
                                        activation=d_hidden_act)
                features = d
                output = tf.layers.dense(features, units=1, kernel_initializer=glorot_uniform, name='D_output')
            return features, output

        ######################
        # GENERATOR FUNCTION #
        ######################
        def generator(condition):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                user_embeddings = tf.get_variable(shape=[self.num_users, num_factors], trainable=True,
                                initializer=glorot_uniform, name='user_embeddings')
                item_embeddings = tf.get_variable(shape=[self.num_items, num_factors], trainable=True,
                                initializer=glorot_uniform, name='item_embeddings')

            user_lookup = tf.nn.embedding_lookup(user_embeddings, condition)
            fake_data = tf.matmul(tf.squeeze(user_lookup, axis=1), item_embeddings, transpose_b=True)
            return fake_data

        self.discriminator, self.generator = discriminator, generator

    def fit(self, num_factors=10, d_layers=1, d_nodes=32, d_hidden_act='linear', epochs=300, batch_size=32, d_lr=1e-4,
            g_lr=1e-4, d_steps=1, g_steps=1, d_reg=0, g_reg=0, recon_coefficient=1e-2, allow_worse=None, freq=None,
            metrics=['MAP'], sample_every=None, validation_evaluator=None, validation_set=None):

        # Construct the model config
        self.config = dict(locals())
        del self.config['self']

        # First clear the session to save GPU memory
        tf.reset_default_graph()
        # Set fixed seed for the TF graph
        tf.set_random_seed(self.seed)

        self.build(num_factors=num_factors, d_layers=d_layers, d_nodes=d_nodes, d_hidden_act=d_hidden_act)

        # Create optimizers
        opt_gen = tf.train.AdamOptimizer(learning_rate=g_lr)
        opt_disc = tf.train.AdamOptimizer(learning_rate=d_lr)

        # placeholders
        real_profile = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        # generator ops
        self.fake_profile = self.generator(self.user_id)

        # discriminator ops
        real_features, real_output = self.discriminator(tf.cast(self.user_id, tf.float32), real_profile)
        fake_features, fake_output = self.discriminator(tf.cast(self.user_id, tf.float32), self.fake_profile)

        # discriminator losses
        loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output))
        loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))

        # model parameters
        self.dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        with tf.variable_scope('dvars_best', reuse=tf.AUTO_REUSE):
            self.dvars_best = []
            for idx, var in enumerate(self.dvars):
                self.dvars_best.append(tf.get_variable('dva_r' + str(idx), shape=var.get_shape(), trainable=False))
        with tf.variable_scope('gvars_best', reuse=tf.AUTO_REUSE):
            self.gvars_best = []
            for idx, var in enumerate(self.gvars):
                self.gvars_best.append(tf.get_variable('gvar_' + str(idx), shape=var.get_shape(), trainable=False))

        # losses
        dloss = loss_real + loss_fake + \
                d_reg * tf.add_n([tf.nn.l2_loss(var) for var in self.dvars])
        gloss = loss_fake + \
                recon_coefficient * tf.losses.mean_squared_error(real_features, fake_features) + \
                g_reg * tf.add_n([tf.nn.l2_loss(var) for var in self.gvars])

        # update ops
        dtrain = opt_disc.minimize(dloss, var_list=self.dvars)
        gtrain = opt_gen.minimize(gloss, var_list=self.gvars)

        ##################
        # START TRAINING #
        ##################
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self._stop_training = False
        if validation_evaluator is not None:
            early_stop = EarlyStoppingScheduler(self, evaluator=validation_evaluator, allow_worse=allow_worse,
                                                freq=freq, metrics=metrics)

        all_users = np.array(range(self.num_users))
        step = batch_size

        train_g_loss = []
        train_d_loss = []

        if self.verbose:
            print('Starting training...')

        t_start = time.time()
        e_start = time.time()

        epoch = 1

        pbar = tqdm.tqdm(total=epochs, initial=1)

        while not self._stop_training and epoch < epochs + 1:
            batch_d_loss = []
            batch_g_loss = []
            np.random.shuffle(all_users)
            for _ in range(d_steps):
                start_idx = 0
                while start_idx < len(all_users):
                    end_idx = start_idx + step
                    if end_idx > len(all_users):
                        end_idx = len(all_users)

                    uids = all_users[start_idx: end_idx]
                    real_histories = self.URM_train[uids].toarray()

                    _, _dloss = self.sess.run([dtrain, dloss],
                                              {real_profile: real_histories, self.user_id: uids.reshape(-1, 1)})
                    batch_d_loss.append(_dloss)
                    start_idx = end_idx

            for _ in range(g_steps):
                start_idx = 0
                while start_idx < len(all_users):
                    end_idx = start_idx + step
                    if end_idx > len(all_users):
                        end_idx = len(all_users)

                    uids = all_users[start_idx: end_idx]
                    real_histories = self.URM_train[uids].toarray()
                    _, _gloss = self.sess.run([gtrain, gloss], {real_profile: real_histories, self.user_id: uids.reshape(-1, 1)})
                    batch_g_loss.append(_gloss)
                    start_idx = end_idx

            mean_epoch_g_loss = np.mean(batch_g_loss)
            mean_epoch_d_loss = np.mean(batch_d_loss)

            train_g_loss.append(mean_epoch_g_loss)
            train_d_loss.append(mean_epoch_d_loss)

            if validation_set is not None and sample_every is not None and epoch % sample_every == 0:
                t_end = time.time()
                total = t_end-e_start
                print('Epoch : {:d}. Total: {:.2f} secs, {:.2f} secs/epoch.'.format(epoch, total, total/sample_every))
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                _, results_run_string = validation_evaluator.evaluateRecommender(self)
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                print(results_run_string)
                e_start = time.time()

            if validation_evaluator is not None:
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                early_stop(epoch)
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()

                if self._stop_training:
                    print('Training stopped, epoch:', epoch)

            epoch += 1
            pbar.update()
        pbar.close()

        t_end = time.time()
        if self.verbose:
            print('Training took {:.2f} seconds'.format(t_end - t_start))

        if self.mode == 'item':
            self.URM_train = self.URM_train.T.tocsr()

        return epoch

    def stop_fit(self):
        self._stop_training = True

    def save_current_model(self):
        # self.saver.save(self.sess, save_path=os.path.join(self.logsdir, 'saved_model', 'model'))
        for idx, var in enumerate(self.dvars):
            self.sess.run(self.dvars_best[idx].assign(var))
        for idx, var in enumerate(self.gvars):
            self.sess.run(self.gvars_best[idx].assign(var))

    def load_model(self):
        # self.saver.restore(self.sess, save_path=os.path.join(self.logsdir, 'saved_model', 'model'))
        # shutil.rmtree(os.path.join(self.logsdir, 'saved_model'), ignore_errors=True)
        for idx, var in enumerate(self.dvars_best):
            self.sess.run(self.dvars[idx].assign(var))
        for idx, var in enumerate(self.gvars_best):
            self.sess.run(self.gvars[idx].assign(var))

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.mode == 'item':
            predictions = self.sess.run(self.fake_profile, {self.user_id: np.array(range(self.num_users)).reshape(-1, 1)})
            return predictions.transpose()[user_id_array]
        else:
            return self.sess.run(self.fake_profile, {self.user_id: user_id_array.reshape(-1, 1)})