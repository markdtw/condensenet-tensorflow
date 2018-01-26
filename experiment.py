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

    def __init__(self, args, is_training=True):
        self.args = args
        self.is_training = is_training
        self.num_examples_train = cifar10.Cifar10DataSet.num_examples_per_epoch(True)
        self.num_examples_eval = cifar10.Cifar10DataSet.num_examples_per_epoch(False)
        self.num_batches_train = self.num_examples_train // self.args.bsize
        self.num_batches_eval = self.num_examples_eval // self.args.bsize
        self.model_dir = './log'

        with tf.device('/cpu:0'):
            with tf.name_scope('cosine_annealing_lr'):
                # cosine learning rate decay
                total_iters = self.args.ep * self.num_batches_train
                self.lr_op = self.args.lr * 0.5 * (1.0 + tf.cos(np.pi * (tf.train.get_or_create_global_step() / total_iters)))

            # momentum optimizer
            self.opt = tf.train.MomentumOptimizer(self.lr_op, 0.9)

            with tf.name_scope('dataset'):
                # get data queue
                self.dataset = cifar10.Cifar10DataSet(self.args.bsize)
                self.dataset.make_batch()

        print ('======= TRAINING CONDENSE NET ==========')
        print ('     Random Seed : {}'.format(self.args.rseed))
        print ('      Initial LR : {}'.format(self.args.lr))
        print ('        LR decay : cosine annealing')
        print ('       Optimizer : Momentum Optimizer')
        print ('        Momentum : 0.9')
        print ('      Batch size : {}'.format(self.args.bsize))
        print ('          Epochs : {}'.format(self.args.ep))
        print (' Num iters in EP : {}'.format(self.num_batches_train))
        print ('  Train Examples : {}'.format(self.num_examples_train))
        print ('    Val Examples : {}'.format(self.num_examples_eval))
        print ('========================================')

    def run(self):

        if self.is_training:    # train and eval
            # build model
            model = CondenseNet(
                num_batches=self.num_batches_train,
                total_ep=self.args.ep,
                stages=[14, 14, 14],
                growth=[8, 16, 32],
                bottleneck=4,
                num_groups=4,
                condense_factor=4,
                num_classes=10
            )

            with slim.arg_scope(cifar_arg_scope()):
                logits, predictions = model.forward(self.dataset.image_batch)

            # calculate loss and compute gradient.
            loss_op = tf.losses.sparse_softmax_cross_entropy(labels=self.dataset.label_batch, logits=logits)
            pdb.set_trace()
            train_op = self.opt.minimize(loss_op, global_step=tf.train.get_or_create_global_step())

            # worker preparation
            summaries_op, sess = self._get_session(loss_op)

            start_time_global = time.time()
            for ep in xrange(self.args.ep):
                # training phase
                sess.run(self.dataset.iterator.initializer, feed_dict={self.dataset.is_training: True})
                start_time_ep = time.time()
                for step in xrange(self.num_batches_train):
                    self._train_step(sess, loss_op, train_op, summaries_op, ep, step)

                # validation phase
                sess.run(self.dataset.iterator.initializer, feed_dict={self.dataset.is_training: False})
                corrects = 0
                for step in xrange(self.num_batches_eval):
                    preds, labels = sess.run([predictions, self.dataset.label_batch])
                    corrects += np.sum(preds == labels)

                # epoch done
                self._logger(mode=1, ep=ep+1, corrects=corrects, elapsed=time.time() - start_time_ep)

            print ('Training done: ' + time.strftime('%M mins: %S secs', time.gmtime(time.time() - start_time_global)))
        else:
            raise NotImplementedError('Evaluation only is currently not supported')

        sess.close()

    def _train_step(self, sess, loss_op, train_op, summaries_op, ep, step):
        """Run training step"""
        start_time = time.time()
        loss, _, summaries = sess.run([loss_op, train_op, summaries_op])
        end_time = time.time()

        # write summaries every iteration
        self.swriter.add_summary(summaries, ep * self.num_batches_train + step)
        # some logs
        if (step + 1) % (self.num_batches_train // 4) == 0:
            self._logger(mode=0, ep=ep+1, step=step+1, loss=loss, elapsed=end_time - start_time)
        if step + 1 == self.num_batches_train:
            self.saver.save(sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=ep+1)

    def _logger(self, mode, **kwargs):

        log = ''
        if mode == 0:       # log training iteration information
            log = 'Epoch %02d (%03d/%03d), loss: %.3f' % (kwargs['ep'], kwargs['step'], self.num_batches_train, kwargs['loss'])
            log_suffix = ' (%.3f sec)' % kwargs['elapsed']
        elif mode == 1:     # log evaluation information
            log = 'Epoch %02d done, evaluation accuracy: %.3f%% (%04d/%04d)' % (kwargs['ep'],\
                    100 * kwargs['corrects'] / (self.num_batches_eval * self.args.bsize),\
                    kwargs['corrects'],\
                    self.num_batches_eval * self.args.bsize)
            log_suffix = ', epoch elapsed: ' + time.strftime('(%M mins: %S secs)', time.gmtime(kwargs['elapsed']))
        else:
            raise NotImplementedError('Logger mode: %d is not supported!' % mode)

        print (log + log_suffix)

    def _get_session(self, loss_op):
        """Saver, configs, variable initializations, summary writer"""
        with tf.device('/cpu:0'):
            # summaries
            tf.summary.scalar('cross_entropy_loss', loss_op)
            tf.summary.scalar('learning_rate', self.lr_op)
            summaries_op = tf.summary.merge_all()

            # savers
            self.saver = []
            self.saver = tf.train.Saver()

            # session configs and session
            config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(
                    force_gpu_compatible=True,
                    allow_growth=True)
            )

            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            if not self.is_training:
                raise NotImplementedError('Restoring variables for evaluation is not yet supported!')

            self.swriter = tf.summary.FileWriter(self.model_dir, sess.graph) if self.is_training else None
            tf.train.start_queue_runners(sess=sess)
            return summaries_op, sess
