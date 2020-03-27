import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from models.Memory import *
from functools import reduce


# kernel_initializer = tf.truncated_normal_initializer(0, 0.02)
# bias_initializer = tf.constant_initializer(0)

class Model(keras.Model):
    def __init__(self, a_dim):
        super().__init__(name='basic_dqn')
        self.l1 = tf.keras.Sequential(
            [
                layers.Conv2D(input_shape=(400, 300, 3), filters=32, kernel_size=(8, 8), strides=(4, 4),
                              padding='valid', activation='relu', kernel_initializer='he_uniform',
                              bias_initializer='zeros'),
                # layers.BatchNormalization()
            ]
        )
        self.l2 = tf.keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu',
                                kernel_initializer='he_uniform', bias_initializer='zeros'),
                # layers.BatchNormalization()
            ]
        )

        self.l3 = tf.keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                                kernel_initializer='he_uniform', bias_initializer='zeros'),
                # layers.BatchNormalization()
            ]
        )
        self.l3_flat = layers.Flatten()
        self.fc1 = tf.keras.Sequential(
            [
                layers.Dense(512, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'),
                # layers.BatchNormalization()
            ]
        )
        self.logits = layers.Dense(a_dim, kernel_initializer='he_uniform', bias_initializer='zeros', name='q_values')

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l3_flat(x)
        x = self.fc1(x)
        x = self.logits(x)
        return x

    def action_value(self, state):
        q_values = self.predict(state, steps=1)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]


class DQNAgent:
    def __init__(self, dqn_para, baskets):
        # self.dqn_para = {
        #     'episode': 300,
        #     'state_dim': [None, 400, 300, 3],
        #     'img_size': (400, 300, 3),
        #     'action_dim': 3,
        #     'lr': 0.00001,
        #     'gamma': 0,
        #     'epsilon': 0.1,
        #     'batch_size': int(5 * len(self.img_dict)),
        #     'buffer_size': int(50 * len(self.img_dict)),
        #     'update_iter': int(100 * len(self.img_dict)),
        #     'start_learning': int(int(10 * len(self.img_dict)))
        # }

        self.action_dim = dqn_para['action_dim']
        self.model = Model(dqn_para['action_dim'])
        self.target_model = Model(dqn_para['action_dim'])
        # print(id(self.model), id(self.target_model))

        opt = optimizers.Adam(learning_rate=dqn_para['lr'])
        self.model.compile(optimizer=opt, loss='mse')

        self.gamma = dqn_para['gamma']
        self.epsilon = dqn_para['epsilon']
        self.batch_size = dqn_para['batch_size']
        self.buffer_size = dqn_para['buffer_size']
        self.num_in_buffer = 0

        self.baskets = baskets
        self.states = np.empty((self.buffer_size,) + dqn_para['img_size'])
        print(np.shape(self.states))
        self.actions = np.empty((self.buffer_size), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size), dtype=np.float32)
        self.dones = np.empty((self.buffer_size), dtype=np.bool)
        self.next_states = np.empty((self.buffer_size,) + dqn_para['img_size'])
        self.next_idx = 0
        self.target_update_iter = dqn_para['update_iter']
        self.start_learning = dqn_para['start_learning']

    def update_actions(self):
        # TODO
        pass

    def train(self, train, test, episode):
        # initialize the initial observation of the agent
        for i_episode in range(1, episode + 1):
            loss = None
            batch_num = 0
            for i, img in enumerate(train):
                state = img['data']
                state = tf.cast(state, tf.float32)
                best_action, q_values = self.model.action_value(state[None])
                action = self.get_action(best_action)  # get the real action

                next_state = train[(i + 1) % len(train)]['data']
                basket_key = list(self.baskets.keys())[action]
                correct_label = img['label']
                reward = 1 if basket_key in correct_label else -1
                done = None

                self.store_transition(state, action, reward, next_state, done)
                self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

                if i > self.start_learning:  # start learning
                    losses = self.train_step()
                    loss = losses if not loss else min(loss, losses)
                    # print('Batch ' + str(batch_num) + 'Loss: ' + str(losses))
                    batch_num += 1
                if i % self.target_update_iter == 0:
                    self.update_target_model()

                # state = None if done else next_state

            train_rewards = self.evaluation(train)
            train_acc = sum(train_rewards == 1) / len(train_rewards)
            test_rewards = self.evaluation(test)
            test_acc = sum(test_rewards == 1) / len(test_rewards)
            print('Episode ' + str(i_episode) + 'Loss: ' + str(loss) + ' Train Accuracy: ' +
                  str(train_acc) + ' Test Accuracy: ' + str(test_acc))

    def train_step(self):
        idxes = self.sample(self.batch_size)
        s_batch = self.states[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]
        done_batch = self.dones[idxes]

        target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis=1) * (1 - done_batch)
        target_f = self.model.predict(s_batch, steps=1)
        for i, val in enumerate(a_batch):
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, target_f)
        # losses = self.model.fit(s_batch, target_f)

        return losses

    def evaluation(self, data):
        # one episode until done
        rewards = []
        for i, img in enumerate(data):
            state = img['data']
            state = tf.cast(state, tf.float32)
            action, q_values = self.model.action_value(state[None])
            act_action, act_q_values = self.target_model.action_value(state[None])
            basket_key = list(self.baskets.keys())[action]
            correct_label = img['label']
            reward = 1 if basket_key in correct_label else -1
            rewards.append(reward)

        return np.array(rewards)

    # store transitions into replay butter
    def store_transition(self, state, action, reward, next_state, done):
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.next_states[self.next_idx] = next_state
        self.dones[self.next_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    # sample n different indexes
    def sample(self, n):
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n:
                break
        return res

    # e-greedy
    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return best_action

    # assign the current network parameters to target network
    def update_target_model(self):
        # print(self.model.get_weights())
        # print(self.target_model.get_weights())
        for i in range(np.shape(self.model.layers)[0]):
            self.target_model.layers[i].set_weights(self.model.layers[i].get_weights())

    def get_target_value(self, state):
        return self.target_model.predict(state, steps=1)

    def e_decay(self):
        self.epsilon *= self.epsilon_decay
