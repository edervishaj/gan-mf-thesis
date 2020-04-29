#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import json
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime

from Base.BaseRecommender import BaseRecommender


class IRGAN(BaseRecommender):
    RECOMMENDER_NAME = 'IRGAN'

    def __init__(self, URM_train, verbose=False):
        self.num_users, self.num_items = URM_train.shape
        self.config = None
        self.verbose = verbose

    def build(self, num_factors=10, init_delta=0.05, disc_reg=1e-4, gen_reg=1e-4):
        self.num_factors = num_factors

        initializer = tf.random_uniform_initializer(minval=-init_delta, maxval=init_delta)

        ##########################
        # DISCRIMINATOR FUNCTION #
        ##########################
        def discriminator(user_id, item_id, label):
            with tf.variable_scope('discrimminator', reuse=tf.AUTO_REUSE):
                user_embeddings = tf.get_variable(shape=[self.num_users, self.num_factors], trainable=True,
                                                  name='disc_user_embedding', initializer=initializer, dtype=tf.float32)

                item_embeddings = tf.get_variable(shape=[self.num_items, self.num_factors], trainable=True,
                                                  name='disc_item_embeddings', initializer=initializer,
                                                  dtype=tf.float32)

                item_bias = tf.get_variable(shape=[self.num_items], trainable=True, name='disc_item_bias')

            user_lookup = tf.nn.embedding_lookup(user_embeddings, user_id)
            item_lookup = tf.nn.embedding_lookup(item_embeddings, item_id)
            item_bias_gather = tf.gather(item_bias, item_id)

            pre_logits = tf.reduce_sum(tf.multiply(user_lookup, item_lookup), 1) + item_bias_gather
            disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pre_logits) + disc_reg * (
                    tf.nn.l2_loss(user_lookup) + tf.nn.l2loss(item_lookup) + tf.nn.l2_loss(item_bias_gather))

            reward_logits = tf.reduce_sum(tf.multiply(user_lookup, item_lookup), 1) + item_bias_gather
            reward = 2 * (tf.sigmoid(reward_logits) - 0.5)

            return disc_loss, reward

        ######################
        # GENERATOR FUNCTION #
        ######################
        def generator(user_id, item_id, reward):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                user_embeddings = tf.get_variable(shape=[self.num_users, self.num_factors], trainable=True,
                                                  name='gen_user_embedding', initializer=initializer, dtype=tf.float32)

                item_embeddings = tf.get_variable(shape=[self.num_items, self.num_factors], trainable=True,
                                                  name='gen_item_embeddings', initializer=initializer, dtype=tf.float32)

                item_bias = tf.get_variable(shape=[self.num_items], trainable=True, name='gen_item_bias')

            user_lookup = tf.nn.embedding_lookup(user_embeddings, user_id)
            item_lookup = tf.nn.embedding_lookup(item_embeddings, item_id)
            item_bias_gather = tf.gather(item_bias, item_id)

            pre_logits = tf.reduce_sum(tf.multiply(user_lookup, item_lookup, 1) + item_bias_gather)
            item_prob = tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(pre_logits, [1, -1])), [-1]), item_id)

            gen_loss = -tf.reduce_mean(tf.log(item_prob) * reward) + gen_reg * (
                    tf.nn.l2_loss(user_lookup) + tf.nn.l2loss(item_lookup) + tf.nn.l2_loss(item_bias_gather))

            predictions_lookup = tf.matmul(user_lookup, item_embeddings, transpose_a=False,
                                           transpose_b=True) + item_bias

            return gen_loss, predictions_lookup

        self.discriminator, self.generator = discriminator, generator

    def fit(self, URM_train, verbose=False, num_factors=10, epochs=500, batch_size=32, D_lr=1e-4, G_lr=1e-4, d_steps=1,
            g_steps=1, disc_reg=1e-4, gen_reg=1e-4, allow_worse=5, sample_every=None, validation_set=None,
            validation_evaluator=None, callback=None):

        # Construct the model config
        self.config = dict(locals())
        self.config['seed'] = self.seed
        del self.config['self']

        # Print config
        if not self.is_experiment:
            with open(os.path.join(self.logsdir, 'config.txt'), 'w') as f:
                json.dump(self.config, f, indent=4)

        # First clear the session to save main/GPU memory
        tf.reset_default_graph()
        # Reset seed of the TF Graph
        tf.set_random_seed(self.seed)

        # Build the discriminator and generator models
        self.build(*URM_train.shape, num_factors, disc_reg, gen_reg)

        # Required for evaluation
        self.URM_train = URM_train

        # Create optimizers
        d_opt = tf.train.GradientDescentOptimizer(learning_rate=D_lr)
        g_opt = tf.train.GradientDescentOptimizer(learning_rate=G_lr)

        # Input placeholders to generator & discriminator
        self.user_id = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        self.item_id = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        label = tf.placeholder(shape=[None, 1], dtype=tf.int32)

        # Discriminator operations
        dloss, reward = self.discriminator(self.user_id, self.item_id, label)

        # Generator operations
        gloss, self.predictions = self.generator(self.user_id, self.item_id, reward)

        # D & G variables
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        # D & G training ops
        D_train_op = d_opt.minimize(dloss, var_list=d_vars)
        G_train_op = g_opt.minimize(gloss, var_list=g_vars)

        init_op = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init_op)

        all_users = np.array(range(self.num_users))

        # START TRAINING
        for epoch in range(1, epochs + 1):
            for _ in range(self.num_users // batch_size):
                # Sample positive and negative (user, item) pairs
                

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        return self.sess.run(self.predictions, feed_dict={self.user_id: user_id_array})
