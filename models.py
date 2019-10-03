from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import numpy as np
import tensorflow as tf
import utils
from time import time

from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class Model:
    def __init__(self, eval_metric, greater_is_better, 
                 epoch, batch_size, verbose, batch_norm=False, dropout_deep=[]):
        self.sess = None
        self.loss = None
        self.optimizer = None

        self.batch_size = batch_size
        self.epoch = epoch
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        
        self.verbose = verbose    
        self.batch_norm = batch_norm
        self.dropout_deep = dropout_deep

        self.train_result, self.valid_result = [], []

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.995, is_training=True, updates_collections=None, reuse=None, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.995, is_training=False, updates_collections=None, reuse=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def fit_on_batch(self, Xi, Xv, y):
        if self.batch_norm:
            if len(self.dropout_deep) == 0:
                feed_dict = {self.feat_index: Xi,
                             self.feat_value: Xv,
                             self.label: y,
                             self.train_phase: True}
            else:
                feed_dict = {self.feat_index: Xi,
                             self.feat_value: Xv,
                             self.label: y,
                             self.train_phase: True,
                             self.dropout_keep_deep: self.dropout_deep}
        else:
            feed_dict = {self.feat_index: Xi,
                        self.feat_value: Xv,
                        self.label: y}
 
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break


    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            if self.batch_norm:
                if len(self.dropout_deep) == 0:
                    feed_dict = {self.feat_index: Xi_batch,
                                 self.feat_value: Xv_batch,
                                 self.label: y_batch,
                                 self.train_phase: False}
                else:
                    feed_dict = {self.feat_index: Xi_batch,
                                 self.feat_value: Xv_batch,
                                 self.label: y_batch,
                                 self.dropout_keep_deep: len(self.dropout_deep) * [1],
                                 self.train_phase: False}
            else:
                feed_dict = {self.feat_index: Xi_batch,
                             self.feat_value: Xv_batch,
                             self.label: y_batch}
 
            batch_out = self.sess.run(self.y_prob, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)

class LR(Model):
    def __init__(self, feature_size, field_size, optimizer_type='gd', learning_rate=1e-2, l2_reg=0, verbose=False,
                 random_seed=None, eval_metric=roc_auc_score, greater_is_better=True, epoch=10, batch_size=1024):
        Model.__init__(self, eval_metric, greater_is_better, epoch, batch_size, verbose)
        init_vars = [('w', [feature_size, 1], 'zero', tf.float32),
                     ('b', [1], 'zero', tf.float32)]
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1

            self.vars = utils.init_var_map(init_vars)

            w = self.vars['w']
            b = self.vars['b']
            self.embeddings = tf.nn.embedding_lookup(w,
                                                    self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)
            
            logits = tf.reduce_sum(self.embeddings, 1) + b
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logits)) + \
                        l2_reg * tf.nn.l2_loss(self.embeddings)
            self.optimizer = utils.get_optimizer(optimizer_type, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

class FM(Model):
    def __init__(self, feature_size, field_size, embedding_size=8, optimizer_type='gd', learning_rate=1e-2, verbose=False,
                 random_seed=None, eval_metric=roc_auc_score, greater_is_better=True, epoch=10, batch_size=1024, l2_w_reg=0, l2_v_reg=0):
        Model.__init__(self, eval_metric, greater_is_better, epoch, batch_size, verbose)
        init_vars = [('w', [feature_size, 1], 'normal', tf.float32),
                     ('b', [1], 'zero', tf.float32),
                     ('v', [feature_size, embedding_size], 'normal', tf.float32)]

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1

            self.vars = utils.init_var_map(init_vars)

            w = self.vars['w']
            b = self.vars['b']
            v = self.vars['v']
            #first order
            self.w_emb = tf.nn.embedding_lookup(w, self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, field_size, 1])
            self.wx = tf.multiply(self.w_emb, feat_value) # None * F * 1
            self.wx_sum = tf.reshape(tf.reduce_sum(self.wx, 1), shape=[-1, 1])
 
            #second order
            self.v_emb = tf.nn.embedding_lookup(v, self.feat_index)
            self.vx = tf.multiply(self.v_emb, feat_value) # None * F * K 

            self.squared_vx = tf.square(self.vx)
            self.sum_squared_vx = tf.reduce_sum(self.squared_vx, 1)

            self.sum_vx = tf.reduce_sum(self.vx, 1)
            self.squared_sum_vx = tf.square(self.sum_vx)

            self.fm_cross_term = tf.reshape(0.5 * tf.reduce_sum(
                tf.subtract(self.squared_sum_vx, self.sum_squared_vx), 1),
                [-1, 1]) # None * 1
            
            logits = self.wx_sum + self.fm_cross_term + b
            self.y_prob = tf.sigmoid(logits)
            
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logits)) + \
                        l2_w_reg * tf.nn.l2_loss(self.w_emb) + l2_v_reg * tf.nn.l2_loss(self.v_emb)
            self.optimizer = utils.get_optimizer(optimizer_type, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

class DeepFM(Model):
    def __init__(self, feature_size, field_size, embedding_size=8, optimizer_type='gd', learning_rate=1e-2, verbose=False,
                 random_seed=None, eval_metric=roc_auc_score, greater_is_better=True, epoch=10, batch_size=1024, l2_reg=0, 
                 deep_layers=[32, 32], batch_norm=True, dropout_deep=[]):
        Model.__init__(self, eval_metric, greater_is_better, epoch, batch_size, verbose, batch_norm, dropout_deep)
        init_vars = [('weight', [feature_size, 1], 'uniform', tf.float32),
                     ('bias', [1], 'uniform', tf.float32),
                     ('feature_embed', [feature_size, embedding_size], 'normal', tf.float32)]
        node_in = embedding_size * field_size 
        for i in range(len(deep_layers)):
            init_vars.extend([('layer_%d' % i, [node_in, deep_layers[i]], 'glorot_normal', tf.float32)])
            init_vars.extend([('bias_%d' % i, [1, deep_layers[i]], 'glorot_normal', tf.float32)])
            node_in = deep_layers[i]

        node_in = field_size + embedding_size + deep_layers[-1]
        init_vars.extend([('concat_projection', [node_in, 1], 'glorot_normal', tf.float32)])
        init_vars.extend([('concat_bias', [1, 1], 'glorot_normal', tf.float32)])
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            
            self.vars = utils.init_var_map(init_vars)

            weight = self.vars['weight']
            bias = self.vars['bias']
            feature_embed = self.vars['feature_embed']
            #first order
            self.w_emb = tf.nn.embedding_lookup(weight, self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, field_size, 1])
            self.wx = tf.reshape(tf.multiply(self.w_emb, feat_value), [-1, field_size]) # None * F
                
            #second order
            self.v_emb = tf.nn.embedding_lookup(feature_embed, self.feat_index)
            self.vx = tf.multiply(self.v_emb, feat_value) # None * F * K 

            self.squared_vx = tf.square(self.vx)
            self.sum_squared_vx = tf.reduce_sum(self.squared_vx, 1)

            self.sum_vx = tf.reduce_sum(self.vx, 1)
            self.squared_sum_vx = tf.square(self.sum_vx)

            self.fm_cross_term = 0.5 * tf.subtract(self.squared_sum_vx, self.sum_squared_vx)
            
            self.y_deep = tf.reshape(self.vx, [-1, embedding_size * field_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(len(deep_layers)):
                self.y_deep = tf.add(
                    tf.matmul(self.y_deep, self.vars['layer_%s' % i]), self.vars['bias_%s' % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%s" % i)
                self.y_deep = tf.nn.dropout(utils.activate(self.y_deep, 'relu'), self.dropout_keep_deep[i+1])

            concat_projection = self.vars['concat_projection']
            concat_bias = self.vars['concat_bias'] 

            self.out = tf.concat([self.wx, self.fm_cross_term, self.y_deep], 1)
            self.out = tf.matmul(self.out, concat_projection) #, concat_bias)
            self.y_prob = tf.sigmoid(self.out)
            
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.out)) + \
                    tf.contrib.layers.l2_regularizer(
                        l2_reg)(self.vars["concat_projection"])
            for i in range(len(deep_layers)):
                self.loss += tf.contrib.layers.l2_regularizer(
                    l2_reg)(self.vars["layer_%d"%i])

            self.optimizer = utils.get_optimizer(optimizer_type, learning_rate, self.loss)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

class PNN(Model):
    def __init__(self, feature_size, field_size, embedding_size=8, optimizer_type='gd', learning_rate=1e-2, verbose=False,
                 random_seed=None, eval_metric=roc_auc_score, greater_is_better=True, epoch=10, batch_size=1024, l2_reg=0, 
                 deep_layers=[32, 32], batch_norm=True, dropout_deep=[]):
        Model.__init__(self, eval_metric, greater_is_better, epoch, batch_size, verbose, batch_norm, dropout_deep)
        init_vars = [('weight', [feature_size, 1], 'uniform', tf.float32),
                     ('bias', [1], 'uniform', tf.float32),
                     ('feature_embed', [feature_size, embedding_size], 'normal', tf.float32)]
        num_pairs = int(field_size * (field_size - 1) / 2)
        node_in = embedding_size * field_size + num_pairs
        for i in range(len(deep_layers)):
            init_vars.extend([('layer_%d' % i, [node_in, deep_layers[i]], 'glorot_normal', tf.float32)])
            init_vars.extend([('bias_%d' % i, [1, deep_layers[i]], 'glorot_normal', tf.float32)])
            node_in = deep_layers[i]

        init_vars.extend([('concat_projection', [deep_layers[-1], 1], 'glorot_normal', tf.float32)])
        init_vars.extend([('concat_bias', [1, 1], 'glorot_normal', tf.float32)])
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            
            self.vars = utils.init_var_map(init_vars)

            self.embeddings = tf.nn.embedding_lookup(self.vars["feature_embed"],
                                                             self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, field_size, 1]) 
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # ---------- Product component --------
            self.xw = tf.reshape(self.embeddings, shape=[-1, field_size * embedding_size])
            #self.xw = tf.nn.dropout(self.xw, self.dropout_keep_pnn[0])

            row = []
            col = []
            for i in range(field_size - 1): 
                for j in range(i+1, field_size):
                    row.append(i)
                    col.append(j)
            p = tf.transpose(tf.gather(tf.transpose(self.embeddings, [1, 0, 2]), row), [1, 0, 2]) 
            q = tf.transpose(tf.gather(tf.transpose(self.embeddings, [1, 0, 2]), col), [1, 0, 2]) 
    
            p = tf.reshape(p, [-1, num_pairs, embedding_size])
            q = tf.reshape(q, [-1, num_pairs, embedding_size])
            self.ip = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])
            self.product_output = tf.concat([self.xw, self.ip], 1)
            #self.product_output = tf.nn.dropout(self.product_output, self.dropout_keep_pnn[1]) 
            # ---------- deep component --------
            self.deep_input_size = num_pairs + field_size * embedding_size 
            self.y_deep = tf.reshape(self.product_output, [-1, self.deep_input_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(len(deep_layers)):
                self.y_deep = tf.add(
                    tf.matmul(self.y_deep, self.vars['layer_%s' % i]), self.vars['bias_%s' % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%s" % i)
                self.y_deep = tf.nn.dropout(utils.activate(self.y_deep, 'relu'), self.dropout_keep_deep[i+1])

            concat_projection = self.vars['concat_projection']
            concat_bias = self.vars['concat_bias'] 

            self.out = tf.add(tf.matmul(self.y_deep, concat_projection), concat_bias)
            self.y_prob = tf.sigmoid(self.out)
            
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.out)) + \
                    tf.contrib.layers.l2_regularizer(
                        l2_reg)(self.vars["concat_projection"])
            for i in range(len(deep_layers)):
                self.loss += tf.contrib.layers.l2_regularizer(
                    l2_reg)(self.vars["layer_%d"%i])

            self.optimizer = utils.get_optimizer(optimizer_type, learning_rate, self.loss)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

class DCN(Model):
    def __init__(self, feature_size, field_size, embedding_size=8, optimizer_type='gd', learning_rate=1e-2, verbose=False,
                 random_seed=None, eval_metric=roc_auc_score, greater_is_better=True, epoch=10, batch_size=1024, l2_reg=0,
                 deep_layers=[32, 32], batch_norm=True, dropout_deep=[], cross_layer_num=3):
        Model.__init__(self, eval_metric, greater_is_better, epoch, batch_size, verbose, batch_norm, dropout_deep)
        init_vars = [('weight', [feature_size, 1], 'uniform', tf.float32),
                     ('bias', [1], 'uniform', tf.float32),
                     ('feature_embed', [feature_size, embedding_size], 'normal', tf.float32)]
    
        node_in = embedding_size * field_size
        for i in range(len(deep_layers)):
            init_vars.extend([('layer_%d' % i, [node_in, deep_layers[i]], 'glorot_normal', tf.float32)])
            init_vars.extend([('bias_%d' % i, [1, deep_layers[i]], 'glorot_normal', tf.float32)])
            node_in = deep_layers[i]
    
        for i in range(cross_layer_num):
            init_vars.extend([('cross_layer_%d' % i, [1, embedding_size * field_size], 'glorot_normal', tf.float32)])
            init_vars.extend([('cross_bias_%d' % i, [1, embedding_size * field_size], 'glorot_normal', tf.float32)])

        node_in = embedding_size * field_size + deep_layers[-1]
        init_vars.extend([('concat_projection', [node_in, 1], 'glorot_normal', tf.float32)])
        init_vars.extend([('concat_bias', [1, 1], 'glorot_normal', tf.float32)])
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
    
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.vars = utils.init_var_map(init_vars)

            self.embeddings = tf.nn.embedding_lookup(self.vars["feature_embed"],
                                                             self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, field_size, 1])
            self.embeddings = tf.reshape(tf.multiply(self.embeddings, feat_value), shape=[-1, embedding_size * field_size])

            # ---------- cross layer ----------
            self.deep_cross_input = tf.nn.dropout(self.embeddings, self.dropout_keep_deep[0])

            self.cross_layer_out = self.deep_cross_input
            for i in range(1, cross_layer_num):
                self.x0xiT = self.deep_cross_input * self.cross_layer_out
                self.x0xiT = tf.reduce_sum(self.x0xiT, 1, keep_dims=True) # None * 1
                self.cross_layer_out = tf.add(tf.matmul(self.x0xiT, self.vars['cross_layer_%d' % i]) \
                    , self.vars['cross_bias_%d' % i]) + self.cross_layer_out

            # ---------- deep component --------
            self.y_deep = self.deep_cross_input
            for i in range(len(deep_layers)):
                self.y_deep = tf.add(
                    tf.matmul(self.y_deep, self.vars['layer_%s' % i]), self.vars['bias_%s' % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%s" % i)
                self.y_deep = tf.nn.dropout(utils.activate(self.y_deep, 'relu'), self.dropout_keep_deep[i+1])

            concat_projection = self.vars['concat_projection']
            concat_bias = self.vars['concat_bias']
            self.out = tf.concat([self.y_deep, self.cross_layer_out], 1)
            self.out = tf.add(tf.matmul(self.out, concat_projection), concat_bias)
            self.y_prob = tf.sigmoid(self.out)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.out)) + \
                    tf.contrib.layers.l2_regularizer(
                        l2_reg)(self.vars["concat_projection"])
            for i in range(len(deep_layers)):
                self.loss += tf.contrib.layers.l2_regularizer(
                    l2_reg)(self.vars["layer_%d"%i])

            self.optimizer = utils.get_optimizer(optimizer_type, learning_rate, self.loss)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

