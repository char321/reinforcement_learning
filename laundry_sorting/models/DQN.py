import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from models.Memory import *
from functools import reduce

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.3),
    'bias_initializer': tf.constant_initializer(0.1)
}
kernel_initializer = tf.truncated_normal_initializer(0, 0.02)
bias_initializer = tf.constant_initializer(0)


class DQN:
    def __init__(self, sess, s_dim, a_dim, batch_size, gamma, lr, epsilon, replace_target_iter):
        self.sess = sess
        self.s_dim = s_dim  # 状态维度
        self.a_dim = a_dim  # one hot行为维度
        self.gamma = gamma
        self.lr = lr  # learning rate
        self.epsilon = epsilon  # epsilon-greedy
        self.replace_target_iter = replace_target_iter  # 经历C步后更新target参数

        self.memory = Memory(batch_size, 10000)
        self._learn_step_counter = 0
        self._generate_model()

    def choose_action(self, s):
        print('choose action')
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.a_dim)
        else:
            q_eval_z = self.sess.run(self.q_eval_z, feed_dict={
                self.s: s[np.newaxis, :]
            })
            return q_eval_z.squeeze().argmax()

    def _generate_model(self):
        print('generate model')
        # self.s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s')
        # self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a')
        # self.r = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        # self.s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_')
        # self.done = tf.placeholder(tf.float32, shape=(None, 1), name='done')

        # self.scalarInput = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        # self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 4032, 3024, 3])

        self.s = tf.placeholder(shape=self.s_dim, dtype=tf.float32, name='s')
        self.s_ = tf.placeholder(shape=self.s_dim, dtype=tf.float32, name='s_')
        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a')
        self.r = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        self.done = tf.placeholder(tf.float32, shape=(None, 1), name='done')
        self.output = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='output')



        self.q_eval_z = self._build_net(self.s, 'eval_net')  # self.s, 'eval_net', True)
        self.q_target_z = self._build_net(self.s_, 'target_net')  # self.s_, 'target_net', False)

        # y = r + gamma * max(q^)
        q_target = self.r + self.gamma * tf.reduce_max(self.q_target_z, axis=1, keepdims=True) * (1 - self.done)

        q_eval = tf.reduce_sum(self.a * self.q_eval_z, axis=1, keepdims=True)
        # a_mask = tf.cast(self.a, tf.bool)
        # q_eval = tf.expand_dims(tf.boolean_mask(self.q_eval_z, a_mask), 1)

        self.loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        param_target = tf.global_variables(scope='target_net')
        param_eval = tf.global_variables(scope='eval_net')

        # 将eval网络参数复制给target网络
        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(param_target, param_eval)]
        print('done')

    def _build_net(self, s, scope):  # , s, scope, trainable):

        print('build_net' + scope)
        with tf.variable_scope(scope):
            #     l = tf.layers.dense(s, 20, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            #     q_z = tf.layers.dense(l, self.a_dim, trainable=trainable, **initializer_helper)
            #
            # return q_z

            conv_filter_w1 = tf.get_variable('w1', shape=[8, 8, 3, 32], initializer=kernel_initializer)
            conv_filter_b1 = tf.get_variable('b1', shape=[32], initializer=bias_initializer)
            self.l1 = tf.nn.relu(tf.nn.conv2d(input=s, filter=conv_filter_w1, strides=[1, 4, 4, 1],
                                              padding='VALID') + conv_filter_b1)

            conv_filter_w2 = tf.get_variable('w2', shape=[4, 4, 32, 64], initializer=kernel_initializer)
            conv_filter_b2 = tf.get_variable('b2', shape=[64], initializer=bias_initializer)
            self.l2 = tf.nn.relu(tf.nn.conv2d(input=self.l1, filter=conv_filter_w2, strides=[1, 2, 2, 1],
                                              padding='VALID') + conv_filter_b2)

            conv_filter_w3 = tf.get_variable('w3', shape=[3, 3, 64, 64], initializer=kernel_initializer)
            conv_filter_b3 = tf.get_variable('b3', shape=[64], initializer=bias_initializer)
            self.l3 = tf.nn.relu(tf.nn.conv2d(input=self.l2, filter=conv_filter_w3, strides=[1, 1, 1, 1],
                                              padding='VALID') + conv_filter_b3)

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            fc_w1 = tf.get_variable('fc_w1', shape=[self.l3_flat.shape[1], 512], initializer=kernel_initializer)
            fc_b1 = tf.get_variable('fc_b1', shape=[512], initializer=bias_initializer)
            self.fc_out1 = tf.nn.relu(tf.matmul(self.l3_flat, fc_w1) + fc_b1)

            q_w = tf.get_variable('q_w', shape=[self.fc_out1.shape[1], self.a_dim], initializer=kernel_initializer)
            q_b = tf.get_variable('q_b', shape=[self.a_dim], initializer=bias_initializer)
            q = tf.matmul(self.fc_out1, q_w) + q_b

        return q

    def store_transition_and_learn(self, s, a, r, s_, done):
        print('store')
        if self._learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_ops)

        # 将行为转换为one hot形式
        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a] = 1

        self.memory.store_transition(s, one_hot_action, [r], s_, [done])
        self._learn()
        self._learn_step_counter += 1

    def _learn(self):
        print('learn')
        s, a, r, s_, done = self.memory.get_mini_batches()

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done
        })
