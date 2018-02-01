# py2 - py3 compatibility settings
from __future__ import absolute_import, division, print_function, unicode_literals
from six import iteritems
from six.moves import xrange
# build-in libraries
import os
import pdb
import time
# pre-installed libraries
import numpy as np
import tensorflow as tf
# local files
import cifar10
from model import CondenseNet, cifar_arg_scope

slim = tf.contrib.slim

class Experiment:

    def __init__(self, args):
        self.args = args
        self.args.stages = list(map(int, self.args.stages.split('-')))
        self.args.growth = list(map(int, self.args.growth.split('-')))

        self.num_examples_train = cifar10.Cifar10DataSet.num_examples_per_epoch(True)
        self.num_examples_eval = cifar10.Cifar10DataSet.num_examples_per_epoch(False)
        self.num_batches_train = self.num_examples_train // self.args.bsize
        self.num_batches_eval = self.num_examples_eval // self.args.bsize

        with tf.device('/cpu:0'):

            with tf.name_scope('dataset'):
                self.dataset = cifar10.Cifar10DataSet(self.args.bsize)
                self.dataset.make_batch()

            with tf.name_scope('cosine_annealing_lr'):
                total_iters = self.args.ep * self.num_batches_train
                lr_op = self.args.lr * 0.5 * (1.0 + tf.cos(np.pi * (tf.train.get_or_create_global_step() / total_iters)))

            self.opt = tf.train.MomentumOptimizer(lr_op, self.args.momentum, use_nesterov=True)

        print ('========= TRAINING CONDENSENET =========')
        print ('      Initial LR : {}'.format(self.args.lr))
        print ('        LR decay : cosine annealing')
        print ('       Optimizer : Momentum Optimizer')
        print ('          Epochs : {}'.format(self.args.ep))
        print ('        Momentum : {}'.format(self.args.momentum))
        print ('          Stages : {}'.format(self.args.stages))
        print ('    Growth Rates : {}'.format(self.args.growth))
        print ('      Batch size : {}'.format(self.args.bsize))
        print ('  Num Batches EP : {}'.format(self.num_batches_train))
        print ('  Train Examples : {}'.format(self.num_examples_train))
        print ('    Val Examples : {}'.format(self.num_batches_eval * self.args.bsize))
        print ('========================================')

    def train(self):

        # build model
        model = CondenseNet(
            num_batches=self.num_batches_train,
            total_ep=self.args.ep,
            stages=self.args.stages,
            growth=self.args.growth,
            bottleneck=4,
            num_groups=4,
            condense_factor=4,
            num_classes=10
        )

        with slim.arg_scope(cifar_arg_scope()):
            logits, predictions = model.forward(self.dataset.image_batch)

        xen_loss_op = tf.losses.sparse_softmax_cross_entropy(labels=self.dataset.label_batch, logits=logits)
        reg_loss_op = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_op = xen_loss_op + tf.add_n(reg_loss_op)

        train_op = self.opt.minimize(loss_op, global_step=tf.train.get_or_create_global_step())

        # worker preparation
        with tf.device('/cpu:0'):
            sess = self._get_session()

        # training phase
        start_time_global = time.time()
        for ep in xrange(self.args.ep):

            sess.run(self.dataset.iterator.initializer, feed_dict={self.dataset.is_training: True})
            for step in xrange(self.num_batches_train):
                loss, _ = sess.run([loss_op, train_op])

                if step % self.args.log_freq == 0:
                    self._logger(mode=0, out=0, ep=ep, step=step, loss=loss)

            # epoch done, save the models
            self.saver.save(sess, os.path.join(self.args.model_dir, 'model.ckpt'), global_step=ep+1)

            # validation phase
            corrects = 0

            sess.run(self.dataset.iterator.initializer, feed_dict={self.dataset.is_training: False})
            for step in xrange(self.num_batches_eval):
                start_time = time.time()
                preds, labels = sess.run([predictions, self.dataset.label_batch])
                elapsed = time.time() - start_time

                corrects += np.sum(preds == labels)

            accuracies = 100 * corrects / (self.num_batches_eval * self.args.bsize)
            self._logger(mode=1, out=0, dur=elapsed, acc=accuracies)

        print ('Training for {} epochs done: '.format(self.args.ep) + \
                time.strftime('%H hrs: %M mins: %S secs', time.gmtime(time.time() - start_time_global)))

        sess.close()

    def _logger(self, mode, out, **kwargs):

        if mode == 0:
            if out == 0:
                print ('EP: %03d (%03d/%03d), LOSS:%.3f' % \
                        (kwargs['ep']+1, kwargs['step']+1, self.num_batches_train, kwargs['loss']))
            else:
                with open(os.path.join(self.args.model_dir, 'training_log.txt'), 'a') as pf:
                    print ('EP: %03d (%03d/%03d), LOSS:%.3f' % \
                            (kwargs['ep']+1, kwargs['step']+1, self.num_batches_train, kwargs['loss']), file=pf)
        else:
            if out == 0:
                print ('EP DONE, ACC: %.3f%%, SECS: %.3f' % (kwargs['acc'], kwargs['dur']))
            else:
                with open(os.path.join(self.args.model_dir, 'validation_log.txt'), 'a') as pf:
                    print ('EP DONE, ACC: %.3f%%, SECS: %.3f' % (kwargs['acc'], kwargs['dur']), file=pf)

        # done

    def _get_session(self):
        """Saver, configs, variable initializations"""
        # savers
        self.saver = tf.train.Saver()

        # config proto and session
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(
                force_gpu_compatible=True,
                allow_growth=True)
        )

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        return sess
