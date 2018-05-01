import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm

# Define the hyper parameters
# The settings below are approximately the same as 20b256f
DEPTH = 64
CADINALITY = 16

RESIDUL_FILTERS = 256
RESIDUL_BLOCKS = 5

LEARNING_RATE = 0.1


def weight_variable(shape):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    return weights


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter,
                                   kernel_size=kernel, strides=stride, padding=padding)
        return network


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(
                           inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Relu(x):
    return tf.nn.relu(x)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


class ResNeXt():
    def __init__(self):
        self.residul_filters = RESIDUL_FILTERS
        self.residul_blocks = RESIDUL_BLOCKS
        self.cadinality = CADINALITY
        self.depth = DEPTH
        self.learning_rate = LEARNING_RATE

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def init(self, dataset, train_iterator, test_iterator):
        # TF variables
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, dataset.output_types, dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = self.session.run(train_iterator.string_handle())
        self.test_handle = self.session.run(test_iterator.string_handle())
        self.init_net(self.next_batch)

    def init_net(self, next_batch):
        # tf.placeholder(tf.float32, [None, 18, 19 * 19])
        self.x = next_batch[0]
        self.y_ = next_batch[1]  # tf.placeholder(tf.float32, [None, 362])
        self.z_ = next_batch[2]  # tf.placeholder(tf.float32, [None, 1])
        # self.batch_norm_count = 0
        self.y_conv, self.z_conv = self.construct_resnext(self.x)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                                       logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        self.mse_loss = \
            tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = \
            tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        loss = 1.0 * self.policy_loss + 1.0 * self.mse_loss + self.reg_term

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt_op = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = \
                opt_op.minimize(loss, global_step=self.global_step)

        correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.avg_policy_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelazlogs/test"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelazlogs/train"), self.session.graph)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session.run(self.init)

    def restore(self, file):
        print("Restoring from {0}".format(file))
        self.saver.restore(self.session, file)

    def process(self, batch_size):
        if not self.time_start:
            self.time_start = time.time()
        # Run training for this batch
        policy_loss, mse_loss, reg_term, _, _ = self.session.run(
            [self.policy_loss, self.mse_loss, self.reg_term, self.train_op,
                self.next_batch],
            feed_dict={self.training: True, self.handle: self.train_handle})
        steps = tf.train.global_step(self.session, self.global_step)
        # Keep running averages
        # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
        # get comparable values.
        mse_loss = mse_loss / 4.0
        self.avg_policy_loss.append(policy_loss)
        self.avg_mse_loss.append(mse_loss)
        self.avg_reg_term.append(reg_term)
        if steps % 1000 == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = batch_size * (1000.0 / elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print("step {}, policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                steps, avg_policy_loss, avg_mse_loss, avg_reg_term,
                # Scale mse_loss back to the original to reflect the actual
                # value being optimized.
                # If you changed the factor in the loss formula above, you need
                # to change it here as well for correct outputs.
                avg_policy_loss + 1.0 * 4.0 * avg_mse_loss + avg_reg_term,
                speed))
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Loss",
                                 simple_value=avg_policy_loss),
                tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
            self.train_writer.add_summary(train_summaries, steps)
            self.time_start = time_end
            self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []
        if steps % 8000 == 0:
            sum_accuracy = 0
            sum_mse = 0
            sum_policy = 0
            test_batches = 800
            for _ in range(0, test_batches):
                test_policy, test_accuracy, test_mse, _ = self.session.run(
                    [self.policy_loss, self.accuracy, self.mse_loss,
                     self.next_batch],
                    feed_dict={self.training: False,
                               self.handle: self.test_handle})
                sum_accuracy += test_accuracy
                sum_mse += test_mse
                sum_policy += test_policy
            sum_accuracy /= test_batches
            sum_policy /= test_batches
            # Additionally rescale to [0, 1] so divide by 4
            sum_mse /= (4.0 * test_batches)
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
            self.test_writer.add_summary(test_summaries, steps)
            print("step {}, policy={:g} training accuracy={:g}%, mse={:g}".
                  format(steps, sum_policy, sum_accuracy * 100.0, sum_mse))
            # save models in a single folder
            path = os.path.join(os.getcwd(), "models/leelaz-model")
            save_path = self.saver.save(self.session, path, global_step=steps)
            print("Model saved in file: {}".format(save_path))
            # leela_path = path + "-" + str(steps) + ".txt"
            # self.save_leelaz_weights(leela_path)
            # print("Leela weights saved to {}".format(leela_path))

    def first_layer(self, x, scope='first_layer'):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=self.residul_filters, kernel=[
                           3, 3], stride=1, layer_name=scope + '_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            return x

    def final_layer(self, x, output_channels, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=output_channels, kernel=[
                           1, 1], stride=1, layer_name=scope + '_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=self.depth, kernel=[
                           1, 1], stride=stride, layer_name=scope + '_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=self.depth, kernel=[
                           3, 3], stride=1, layer_name=scope + '_conv2')
            x = Batch_Normalization(
                x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(CADINALITY):
                splits = self.transform_layer(
                    input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)
            return Concatenation(layers_split)

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=self.residul_filters, kernel=[
                           1, 1], stride=1, layer_name=scope + '_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope + '_batch1')
            return x

    def residul_layer(self, input_x, layer_num):
        x = self.split_layer(
            input_x, stride=1, layer_name='split_layer_' + str(layer_num))
        x = self.transition_layer(x, scope='trans_layer_' + str(layer_num))
        return Relu(x + input_x)

    def construct_resnext(self, planes):
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])
        # NCHW -> NHWC
        x_planes = tf.transpose(x_planes, [0, 2, 3, 1])

        x = self.first_layer(x_planes)

        for i in range(self.residul_blocks):
            x = self.residul_layer(x, i)

        # Policy head
        conv_pol = self.final_layer(x, output_channels=2, scope='policy')
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2 * 19 * 19])
        W_fc1 = weight_variable([2 * 19 * 19, (19 * 19) + 1])
        b_fc1 = bias_variable([(19 * 19) + 1])
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head
        conv_val = self.final_layer(x, output_channels=1, scope='value')
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19 * 19])
        W_fc2 = weight_variable([19 * 19, 256])
        b_fc2 = bias_variable([256])
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable([256, 1])
        b_fc3 = bias_variable([1])
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        return h_fc1, h_fc3
