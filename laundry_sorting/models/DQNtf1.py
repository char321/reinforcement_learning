import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from models.Memory import *
from functools import reduce

kernel_initializer = tf.truncated_normal_initializer(0, 0.02)
bias_initializer = tf.constant_initializer(0)


class DQNtf1:
    def __init__(self, sess, dqn_para, is_test=False):
        self.sess = sess
        if is_test:
            self.epsilon = 0
        else:
            self.s_dim = dqn_para['state_dim'],
            self.a_dim = dqn_para['action_dim'],
            self.batch_size = dqn_para['batch_size'],
            self.buffer_size = dqn_para['buffer_size'],
            self.gamma = dqn_para['gamma'],
            self.lr = dqn_para['lr'],
            self.epsilon = dqn_para['epsilon'],
            self.replace_target_iter = dqn_para['update_iter']
            self.memory = Memory(dqn_para['batch_size'], dqn_para['buffer_size'])
            self.learn_step_counter = 0
            self.update_time = 0

        if self.s_dim:
            self.generate_model()

    def generate_model(self):
        print('generate model')

        # input of evaluate net
        self.s = tf.placeholder(shape=self.s_dim, dtype=tf.float32, name='s')  # state
        # input of target net
        self.s_ = tf.placeholder(shape=self.s_dim, dtype=tf.float32, name='s_')  # state
        # output q-valur of action
        self.a = tf.placeholder(tf.float32, shape=(None, None), name='a')  # action
        self.r = tf.placeholder(tf.float32, shape=(None, 1), name='r')  # reward
        self.done = tf.placeholder(tf.float32, shape=(None, 1), name='done')  # indicate if the sorting behaviour is end

        # evaluate net
        self.q_eval_z = self.build_net(self.s, 'eval_net_' + str(self.update_time))
        # target net
        self.q_target_z = self.build_net(self.s_, 'target_net_' + str(self.update_time))

        # y = r + gamma * max(q^)
        self.q_target = self.r + self.gamma * tf.reduce_max(self.q_target_z, axis=1, keepdims=True) * (1 - self.done)

        self.q_eval = tf.reduce_sum(self.a * self.q_eval_z, axis=1, keepdims=True)
        # a_mask = tf.cast(self.a, tf.bool)
        # q_eval = tf.expand_dims(tf.boolean_mask(self.q_eval_z, a_mask), 1)

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        param_target = tf.global_variables(scope='target_net_' + str(self.update_time))
        param_eval = tf.global_variables(scope='eval_net_' + str(self.update_time))

        # define the copy operation
        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(param_target, param_eval)]

    def build_net(self, s, scope):  # , s, scope, trainable):

        print('build_net' + scope)
        with tf.variable_scope(scope):
            self.conv_filter_w1 = tf.get_variable('w1', shape=[8, 8, 3, 32], initializer=kernel_initializer)
            self.conv_filter_b1 = tf.get_variable('b1', shape=[32], initializer=bias_initializer)
            self.l1 = tf.nn.relu(tf.nn.conv2d(input=s, filter=self.conv_filter_w1, strides=[1, 4, 4, 1],
                                              padding='VALID') + self.conv_filter_b1)

            self.conv_filter_w2 = tf.get_variable('w2', shape=[4, 4, 32, 64], initializer=kernel_initializer)
            self.conv_filter_b2 = tf.get_variable('b2', shape=[64], initializer=bias_initializer)
            self.l2 = tf.nn.relu(tf.nn.conv2d(input=self.l1, filter=self.conv_filter_w2, strides=[1, 2, 2, 1],
                                              padding='VALID') + self.conv_filter_b2)

            self.conv_filter_w3 = tf.get_variable('w3', shape=[3, 3, 64, 64], initializer=kernel_initializer)
            self.conv_filter_b3 = tf.get_variable('b3', shape=[64], initializer=bias_initializer)
            self.l3 = tf.nn.relu(tf.nn.conv2d(input=self.l2, filter=self.conv_filter_w3, strides=[1, 1, 1, 1],
                                              padding='VALID') + self.conv_filter_b3)

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            self.fc_w1 = tf.get_variable('fc_w1', shape=[self.l3_flat.shape[1], 512], initializer=kernel_initializer)
            self.fc_b1 = tf.get_variable('fc_b1', shape=[512], initializer=bias_initializer)
            self.fc_out1 = tf.nn.relu(tf.matmul(self.l3_flat, self.fc_w1) + self.fc_b1)

            self.q_w = tf.get_variable('q_w', shape=[self.fc_out1.shape[1], self.a_dim], initializer=kernel_initializer)
            self.q_b = tf.get_variable('q_b', shape=[self.a_dim], initializer=bias_initializer)

            # self.q = tf.get_variable('q_out', shape=[self.a_dim])
            self.matmul = tf.matmul(self.fc_out1, self.q_w, name='matmul')
            self.q = self.matmul + self.q_b

        return self.q

    def update_actions(self):
        self.update_time += 1
        self.a_dim += 1
        self.memory.increase_action_dim()

        # evaluate net
        self.q_eval_z = self.update_net(self.s, 'eval_net_' + str(self.update_time))
        # target net
        self.q_target_z = self.update_net(self.s_, 'target_net_' + str(self.update_time))

        # y = r + gamma * max(q^)
        self.q_target = self.r + self.gamma * tf.reduce_max(self.q_target_z, axis=1, keepdims=True) * (1 - self.done)
        self.q_eval = tf.reduce_sum(self.a * self.q_eval_z, axis=1, keepdims=True)
        # a_mask = tf.cast(self.a, tf.bool)
        # q_eval = tf.expand_dims(tf.boolean_mask(self.q_eval_z, a_mask), 1)

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self.optimizer = tf.train.AdamOptimizer(self.lr, name='opt').minimize(self.loss)

        param_target = tf.global_variables(scope='target_net_' + str(self.update_time))
        param_eval = tf.global_variables(scope='eval_net_' + str(self.update_time))

        # redefine the copy operation
        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(param_target, param_eval)]

    # update the CNN net using new action dimension
    def update_net(self, s, scope):
        print('update_net' + scope)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            new_w1 = tf.get_variable('w1', shape=[8, 8, 3, 32])
            self.conv_filter_w1 = tf.assign(new_w1, self.conv_filter_w1)
            new_b1 = tf.get_variable('b1', shape=[32])
            self.conv_filter_b1 = tf.assign(new_b1, self.conv_filter_b1)
            self.l1 = tf.nn.relu(tf.nn.conv2d(input=s, filter=self.conv_filter_w1, strides=[1, 4, 4, 1],
                                              padding='VALID') + self.conv_filter_b1)

            new_w2 = tf.get_variable('w2', shape=[4, 4, 32, 64])
            self.conv_filter_w2 = tf.assign(new_w2, self.conv_filter_w2)
            new_b2 = tf.get_variable('b2', shape=[64])
            self.conv_filter_b2 = tf.assign(new_b2, self.conv_filter_b2)
            self.l2 = tf.nn.relu(tf.nn.conv2d(input=self.l1, filter=self.conv_filter_w2, strides=[1, 2, 2, 1],
                                              padding='VALID') + self.conv_filter_b2)

            new_w3 = tf.get_variable('w3', shape=[3, 3, 64, 64])
            self.conv_filter_w3 = tf.assign(new_w3, self.conv_filter_w3)
            new_b3 = tf.get_variable('b3', shape=[64])
            self.conv_filter_b3 = tf.assign(new_b3, self.conv_filter_b3)
            self.l3 = tf.nn.relu(tf.nn.conv2d(input=self.l2, filter=self.conv_filter_w3, strides=[1, 1, 1, 1],
                                              padding='VALID') + self.conv_filter_b3)

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            new_fc_w1 = tf.get_variable('fc_w1', shape=[self.l3_flat.shape[1], 512])
            self.fc_w1 = tf.assign(new_fc_w1, self.fc_w1)
            new_fc_b1 = tf.get_variable('fc_b1', shape=[512])
            self.fc_b1 = tf.assign(new_fc_b1, self.fc_b1)
            self.fc_out1 = tf.nn.relu(tf.matmul(self.l3_flat, self.fc_w1) + self.fc_b1)

            self.q_w = tf.Variable(
                tf.truncated_normal(shape=[self.fc_out1.shape[1], self.a_dim], stddev=0.02, dtype=tf.float32),
                name='q_w')
            self.q_b = tf.Variable(tf.constant(0, shape=[self.a_dim], dtype=tf.float32), name='q_b')

            self.matmul = tf.matmul(self.fc_out1, self.q_w, name='matmul')
            self.q = self.matmul + self.q_b

        return self.q

    def store_transition_and_learn(self, s, a, r, s_, done):
        # print('store')
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_ops)

        # change to one-hot representation
        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a] = 1

        self.memory.store_transition(s, one_hot_action, [r], s_, [done])
        self.learn()
        self.learn_step_counter += 1

    def choose_action(self, s):

        # print('choose action')
        # print(self.q_eval_z)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.a_dim)
        else:
            q_eval_z = self.sess.run(self.q_eval_z, feed_dict={
                self.s: s[np.newaxis, :]
            })
            return q_eval_z.squeeze().argmax()

    def learn(self):
        # print('learn')
        s, a, r, s_, done = self.memory.get_mini_batches()

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done
        })

    def reload_tensor(self, update_time):
        self.update_time = update_time
        graph = tf.get_default_graph()
        self.s = graph.get_tensor_by_name('s:0')
        self.q_eval_z = graph.get_tensor_by_name('eval_net_' + str(update_time) + '/q_b:0') + \
                        graph.get_tensor_by_name('eval_net_' + str(update_time) + '/matmul_1:0')

    def predict(self, s):
        q_eval_z = self.sess.run(self.q_eval_z, feed_dict={
            self.s: s[np.newaxis, :]
        })
        return q_eval_z.squeeze().argmax()

    def copy_net(self):
        self.sess.run(self.target_replace_ops)