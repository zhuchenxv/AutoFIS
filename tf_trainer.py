from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
from grda_tensorflow import GRDA
from tf_utils import get_optimizer, get_loss


class Trainer:
    logdir = None
    session = None
    dataset = None
    model = None
    saver = None
    learning_rate = None
    train_pos_ratio = None
    test_pos_ratio = None
    ckpt_time = None

    def __init__(self, model=None, train_gen=None, test_gen=None, valid_gen=None,
                 opt1='adam', opt2='grda', epsilon=1e-8, initial_accumulator_value=1e-8, momentum=0.95,
                 loss='weighted', pos_weight=1.0,
                 n_epoch=1, train_per_epoch=10000, test_per_epoch=10000, early_stop_epoch=5,
                 batch_size=2000, learning_rate=1e-2, decay_rate=0.95, learning_rate2=1e-2,decay_rate2=1,
                 logdir=None, load_ckpt=False, ckpt_time=10,grda_c=0.005, grda_mu=0.51,
                 test_every_epoch=1, retrain_stage=0):
        self.model = model
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.valid_gen = valid_gen
        optimizer = get_optimizer(opt1)
        loss = get_loss(loss)
        self.pos_weight = pos_weight
        self.n_epoch = n_epoch
        self.train_per_epoch = train_per_epoch + 1
        self.early_stop_epoch = early_stop_epoch
        self.test_per_epoch = test_per_epoch
        self.batch_size = batch_size
        self._learning_rate = learning_rate
        self.decay_rate = decay_rate
        self._learning_rate2 = learning_rate2
        self.decay_rate2 = decay_rate2
        self.logdir = logdir
        self.ckpt_time = ckpt_time
        self.epsilon = epsilon
        self.test_every_epoch = test_every_epoch
        self.retrain_stage = retrain_stage

        self.call_auc = roc_auc_score
        self.call_loss = log_loss
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                # device_count={'GPU': 0},
                                )
        config.gpu_options.allow_growth = True
        # config.log_device_placement=True
        self.session = tf.Session(config=config)

        self.learning_rate = tf.placeholder("float")
        self.learning_rate2 = tf.placeholder("float")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        tf.summary.scalar('global_step', self.global_step)
        if opt1 == 'adam':
            opt1 = optimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)  # TODO fbh
        elif opt1 == 'adagrad':
            opt1 = optimizer(learning_rate=self.learning_rate, initial_accumulator_value=initial_accumulator_value)
        elif opt1 == 'moment':
            opt1 = optimizer(learning_rate=self.learning_rate, momentum=momentum)
        elif opt1 == 'grda':
            opt1 = GRDA(learning_rate=self.learning_rate, c=grda_c, mu=grda_mu)
        else:
            opt1 = optimizer(learning_rate=self.learning_rate, )  # TODO fbh

        if opt2 == 'grda':
            opt2 = GRDA(learning_rate=self.learning_rate2, c=grda_c, mu=grda_mu)
        self.model.compile(loss=loss, optimizer1=opt1, optimizer2=opt2,global_step=self.global_step, pos_weight=pos_weight)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def _run(self, fetches, feed_dict):
        return self.session.run(fetches=fetches, feed_dict=feed_dict)

    def _train(self, X, y):
        feed_dict = {
            self.model.labels: y,
            self.learning_rate: self._learning_rate,
            self.learning_rate2: self._learning_rate2
        }
        if type(self.model.inputs) is list:
            for i in range(len(self.model.inputs)):
                feed_dict[self.model.inputs[i]] = X[i]
        else:
            feed_dict[self.model.inputs] = X
        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = True
        if self.model.l2_loss is None:
            if self.retrain_stage:
                _, _loss, outputs = self._run(fetches=[self.model.optimizer1, self.model.loss, self.model.outputs],feed_dict=feed_dict)
                _l2_loss = 0
            else:
                _, _, _loss, outputs = self._run(fetches=[self.model.optimizer1, self.model.optimizer2, self.model.loss, self.model.outputs],feed_dict=feed_dict)
                _l2_loss = 0
        else:
            if self.retrain_stage:
                _, _loss, _l2_loss, outputs = self._run(fetches=[self.model.optimizer1, self.model.loss, self.model.l2_loss, self.model.outputs], feed_dict=feed_dict)
            else:
                _, _, _loss, _l2_loss, outputs = self._run(fetches=[self.model.optimizer1, self.model.optimizer2, self.model.loss, self.model.l2_loss, self.model.outputs], feed_dict=feed_dict)
        return _loss, _l2_loss, outputs

    def _watch(self, X, y, training, watch_list):
        feed_dict = {
            self.model.labels: y,
            self.learning_rate: self._learning_rate,
            self.learning_rate2: self._learning_rate2,
        }
        if type(self.model.inputs) is list:
            for i in range(len(self.model.inputs)):
                feed_dict[self.model.inputs[i]] = X[i]
        else:
            feed_dict[self.model.inputs] = X
        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = training
        if self.retrain_stage:
            fetches = [self.model.optimizer1, self.model.loss]
        else:
            fetches = [self.model.optimizer1, self.model.optimizer2, self.model.loss]
        fetches.extend(watch_list)
        return self._run(fetches=fetches, feed_dict=feed_dict)

    def _predict(self, X, y):
        feed_dict = {
            self.model.labels: y
        }
        if type(self.model.inputs) is list:
            for i in range(len(self.model.inputs)):
                feed_dict[self.model.inputs[i]] = X[i]
        else:
            feed_dict[self.model.inputs] = X
        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = False
        return self._run(fetches=[self.model.loss, self.model.outputs], feed_dict=feed_dict)


    def predict(self, gen, eval_size):
        preds = []
        labels = []
        cnt = 0
        tic = time.time()
        num = 0
        for batch_data in gen:
            X, y = batch_data
            batch_loss, batch_pred = self._predict(X, y)
            preds.append(batch_pred)
            labels.append(y)
            cnt += 1
            if cnt % 100 == 0:
                print('evaluated batches:', cnt, time.time() - tic)
                tic = time.time()
            num += 1
            if num >= int(self.test_per_epoch/self.batch_size):
                break
        preds = np.concatenate(preds)
        preds = np.float64(preds)
        preds = np.clip(preds, 1e-8, 1 - 1e-8)
        labels = np.concatenate(labels)
        loss = self.call_loss(y_true=labels, y_pred=preds)
        auc = self.call_auc(y_score=preds, y_true=labels)
        return labels, preds, loss, auc


    def _batch_callback(self):
        pass

    def _epoch_callback(self,):
        tic = time.time()
        print('running test...')
        labels, preds, loss, auc = self.predict(self.test_gen, self.test_per_epoch)
        print('test loss = %f, test auc = %f' % (loss, auc))
        toc = time.time()
        print('evaluated time:', str(datetime.timedelta(seconds=int(toc - tic))))
        print("analyse_structure")
        self.model.analyse_structure(self.session, print_full_weight=True)
        return loss, auc

    def score(self):
        self._epoch_callback()

    def fit(self):
        self.model.analyse_structure(self.session, print_full_weight=False)
        num_of_batches = int(np.ceil(self.train_per_epoch / self.batch_size))
        total_batches = self.n_epoch * num_of_batches
        print('total batches: %d\tbatch per epoch: %d' % (total_batches, num_of_batches))
        start_time = time.time()
        tic = time.time()
        epoch = 1
        finished_batches = 0
        avg_loss = 0
        avg_l2 = 0
        label_list = []
        pred_list = []
        tx = []
        loss_list = []
        auc_list = []
        last_epoch = -1
        train_opt = 1

        test_every_epoch = self.test_every_epoch

        while epoch <= self.n_epoch:
            print('new iteration')
            epoch_batches = 0

            for batch_data in self.train_gen:
                X, y = batch_data
                label_list.append(y)
                if last_epoch != epoch:
                    last_epoch = epoch
                batch_loss, batch_l2, batch_pred = self._train(X, y)


                pred_list.append(batch_pred)
                avg_loss += batch_loss
                avg_l2 += batch_l2
                finished_batches += 1
                epoch_batches += 1

                epoch_batch_num = 100
                if epoch_batches % epoch_batch_num == 0:
                    avg_loss /= epoch_batch_num
                    avg_l2 /= epoch_batch_num
                    label_list = np.concatenate(label_list)
                    pred_list = np.concatenate(pred_list)
                    moving_auc = self.call_auc(y_true=label_list, y_score=pred_list)
                    elapsed = int(time.time() - start_time)
                    eta = int((total_batches - finished_batches) / finished_batches * elapsed)
                    print("elapsed : %s, ETA : %s" % (str(datetime.timedelta(seconds=elapsed)),
                                                      str(datetime.timedelta(seconds=eta))))
                    print('epoch %d / %d, batch %d / %d, global_step = %d, learning_rate = %e, loss = %f, l2 = %f, '
                          'auc = %f' % (epoch, self.n_epoch, epoch_batches, num_of_batches,
                                        self.global_step.eval(self.session), self._learning_rate,
                                        avg_loss, avg_l2, moving_auc))
                    label_list = []
                    pred_list = []
                    avg_loss = 0
                    avg_l2 = 0

                toc = time.time()
                if toc - tic > self.ckpt_time * 60:
                    tic = toc

                if epoch_batches % num_of_batches == 0:
                    if epoch % test_every_epoch == 0:
                        l, a = self._epoch_callback()
                        loss_list.append(l)
                        auc_list.append(a)
                    self._learning_rate *= self.decay_rate
                    self._learning_rate2 *= self.decay_rate2
                    epoch += 1
                    epoch_batches = 0
                    if epoch > self.n_epoch:
                        return

            if epoch_batches % num_of_batches != 0:
                if epoch % test_every_epoch == 0:
                    l, a = self._epoch_callback()
                    loss_list.append(l)
                    auc_list.append(a)
                self._learning_rate *= self.decay_rate
                self._learning_rate2 *= self.decay_rate2
                epoch += 1
                epoch_batches = 0
                if epoch > self.n_epoch:
                    return

