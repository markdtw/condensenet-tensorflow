# py2 - py3 compatibility settings
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import xrange
# build-in libraries
import os
import pdb
# installed libraries
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def cifar_arg_scope(weight_decay=1e-4):
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    batch_norm_params = {
        'decay': 0.9,
        'epsilon': 1e-5,
        'activation_fn': tf.nn.relu,
        'fused': True
    }
    variable_params = {
        'initializer': weights_initializer,
        'regularizer': weights_regularizer
    }
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None,
                        #biases_initializer=None,
                        weights_initializer=weights_initializer,
                        weights_regularizer=weights_regularizer):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.variable], **variable_params) as sc:
                return sc

class CondenseNet:

    def __init__(self,
            num_batches, total_ep,
            stages=[14, 14, 14],
            growth=[8, 16, 32],
            bottleneck=4,
            num_groups=4,
            condense_factor=4,
            num_classes=10):

        self.num_batches = num_batches
        self.total_ep = total_ep
        self.stages = stages
        self.growth = growth
        self.bottleneck = bottleneck
        self.num_groups = num_groups
        self.condense_factor = condense_factor
        self.num_classes = num_classes

        self.n_stages = len(self.stages)

    def forward(self, inputs):
        # Init conv (stem)
        net = slim.conv2d(inputs, num_outputs=2 * self.growth[0], kernel_size=3, scope='init_conv')    # (n, h, w, 16)

        # Stages
        for i in xrange(self.n_stages):
            with tf.variable_scope('block_{}'.format(i)):
                net = self.dense_block(net, i)

        net = tf.squeeze(net, axis=[1, 2])
        # Classifier
        with tf.variable_scope('classifier'):
            logits = slim.fully_connected(net, self.num_classes, activation_fn=None)
            predictions = tf.argmax(logits, axis=-1)
        return logits, predictions

    def dense_block(self, net, i):

        last = (i == self.n_stages - 1)

        for j in xrange(self.stages[i]):
            curr_net = net
            net = self.dense_layer(net, self.growth[i], j)
            net = tf.concat([net, curr_net], axis=-1)

        if not last:
            net = slim.avg_pool2d(net, 2, scope='avg_pool')
        else:
            n, h, w, c = net.shape
            net = slim.batch_norm(net, scope='bn-relu')
            net = slim.avg_pool2d(net, [h, w], stride=[h, w], scope='global-avg_pool')

        return net

    def dense_layer(self, net, growth_rate, j):

        # 1x1 learned group conv
        net = slim.batch_norm(net, scope='bn-relu-lgc-{}'.format(j))
        net = self.learned_group_conv(net, self.bottleneck * growth_rate, scope='lgc-{}'.format(j))
        #net = slim.conv2d(net, self.bottleneck * growth_rate, 1, scope='1x1conv-{}'.format(j))

        # 3x3 standard group conv
        net = slim.batch_norm(net, scope='bn-relu-sgc-{}'.format(j))
        net = self.standard_group_conv(net, growth_rate, scope='sgc-{}'.format(j))
        #net = slim.conv2d(net, growth_rate, 3, scope='3x3conv-{}'.format(j))

        return net

    def learned_group_conv(self, net, num_outputs, kernel_size=1, scope='lgc'):
        # 1x1 learned group conv
        _, _, _, num_inputs = net.shape
        curr_iter = tf.train.get_or_create_global_step()
        condensing_stages = self.num_batches * self.total_ep // (2 * (self.condense_factor - 1))
        stage = curr_iter+1 // condensing_stages

        with tf.variable_scope(scope):
            conv_weights = slim.variable('weights', shape=[1, 1, num_inputs, num_outputs])
            mask = tf.get_variable('mask', shape=conv_weights.shape, trainable=False, initializer=tf.constant_initializer(1))

            def pruning():

                weights = tf.squeeze(tf.abs(conv_weights * mask))
                w_in, w_out = weights.shape
                d_out = int(w_out // self.num_groups)
                delta = int(w_in // self.condense_factor)

                # shuffle
                weights = tf.reshape(weights, [w_in, d_out, self.num_groups])
                weights = tf.transpose(weights, [0, 2, 1])
                weights = tf.reshape(weights, [w_in, w_out])

                control_ops = []
                for i in xrange(self.num_groups):
                    wi = weights[:, i * d_out:(i+1) * d_out]  # (w_in, d_out)
                    wis = tf.reduce_sum(wi, axis=-1)    # (w_in,)
                    _, indices = tf.nn.top_k(wis * -1, k=tf.cast(stage*delta, tf.int32))

                    for j in xrange(delta):
                        indice = indices[tf.cast((stage-1)*delta + j, tf.int32)]
                        op = mask[:, :, indice, i::self.num_groups].assign(np.zeros([1, 1, d_out]))
                        control_ops.append(op)

                with tf.control_dependencies(control_ops):
                    return tf.identity(mask)

            #mask = tf.cond(tf.logical_or(tf.equal(stage, 1.0), tf.equal(stage, 2.0)),
            #    true_fn=pruning, false_fn=lambda: tf.identity(mask)
            #)
            mask = tf.cond(tf.logical_and(
                    tf.equal(tf.mod(curr_iter+1, condensing_stages), 0),
                    tf.less(curr_iter+1 // condensing_stages, self.condense_factor)
                ), true_fn=pruning, false_fn=lambda: tf.identity(mask)
            )

            masked_conv_weights = conv_weights * mask
            net = tf.nn.conv2d(net, masked_conv_weights, [1, 1, 1, 1], 'SAME')

        return net

    def standard_group_conv(self, net, num_outputs, kernel_size=3, scope='sgc'):

        n, h, w, c = net.shape

        with tf.variable_scope(scope):
            net_splits = tf.split(net, [int(c // self.num_groups)] * self.num_groups, axis=-1)
            net = [slim.conv2d(net_split, num_outputs // self.num_groups, kernel_size) for net_split in net_splits]
            net = tf.concat(net, axis=-1)   # (n, h, w, num_outputs)

        return net
