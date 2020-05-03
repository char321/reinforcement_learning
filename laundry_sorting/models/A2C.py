# import gym
# import logging
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import tensorflow.keras.layers as kl
# import tensorflow.keras.losses as kls
# import tensorflow.keras.optimizers as ko

import gym
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import losses
from components.Config import Config
from models.Memory import *

config = Config()
kernel_initializer = config.dqn_para['initializer'][0]
bias_initializer = config.dqn_para['initializer'][1]


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(keras.Model):
    def __init__(self, a_dim):
        super().__init__(name='basic_dqn_' + str(a_dim))
        self.a_dim = a_dim
        self.l1 = tf.keras.Sequential(
            [
                layers.Conv2D(input_shape=(400, 300, 3), filters=32, kernel_size=(8, 8), strides=(4, 4),
                              padding='valid', activation='relu', kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )
        self.l2 = tf.keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu',
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )

        self.l3 = tf.keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )
        self.l3_flat = layers.Flatten()
        self.fc1 = tf.keras.Sequential(
            [
                layers.Dense(512, activation='relu', kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )
        self.logits = layers.Dense(a_dim, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                   name='q_values')

        self.vl1 = tf.keras.Sequential(
            [
                layers.Conv2D(input_shape=(400, 300, 3), filters=32, kernel_size=(8, 8), strides=(4, 4),
                              padding='valid', activation='relu', kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )
        self.vl2 = tf.keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu',
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )

        self.vl3 = tf.keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )
        self.vl3_flat = layers.Flatten()
        self.vfc1 = tf.keras.Sequential(
            [
                layers.Dense(512, activation='relu', kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer),
                # layers.BatchNormalization()
            ]
        )
        self.value = layers.Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                   name='q_values')

        self.dist = ProbabilityDistribution()

    # def build(self, inputs_shape):

    # def __init__(self, num_actions):
    #     super().__init__('mlp_policy')
    #     # no tf.get_variable(), just simple Keras API
    #     self.hidden1 = kl.Dense(128, activation='relu')
    #     self.hidden2 = kl.Dense(128, activation='relu')
    #     self.value = kl.Dense(1, name='value')
    #     # logits are unnormalized log probabilities
    #     self.logits = kl.Dense(num_actions, name='policy_logits')
    #     self.dist = ProbabilityDistribution()
    #
    # def call(self, inputs):
    #     # inputs is a numpy array, convert to Tensor
    #     x = tf.convert_to_tensor(inputs)
    #     # separate hidden layers from the same input tensor
    #     hidden_logs = self.hidden1(x)
    #     hidden_vals = self.hidden2(x)
    #     return self.logits(hidden_logs), self.value(hidden_vals)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l3_flat(x)
        x = self.fc1(x)
        logits = self.logits(x)

        x = self.vl1(inputs)
        x = self.vl2(x)
        x = self.vl3(x)
        x = self.vl3_flat(x)
        x = self.vfc1(x)
        values = self.value(x)
        return logits, values

    def action_value(self, state):
        # executes call() under the hood
        logits, value = self.predict(state)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        # print('logits' + str(logits))
        # print('action' + str(action))
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, dqn_para, baskets):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.gamma = dqn_para['gamma']
        self.value = 0.5
        self.entropy = 0.0001

        # self.params = {
        #     'gamma': 0.99,
        #     'value': 0.5,
        #     'entropy': 0.0001
        # }

        self.dqn_para = dqn_para.copy()
        self.action_dim = dqn_para['action_dim']
        self.model = Model(self.action_dim)
        self.batch_size = dqn_para['batch_size']
        self.baskets = baskets

        opt = Adam(learning_rate=dqn_para['lr'])
        if config.dqn_para['optimizer'] == 'SGD':
            opt = SGD(learning_rate=dqn_para['lr'], momentum=dqn_para['momentum'])
        self.model.compile(optimizer=opt, loss='mse')

        self.model.compile(
            optimizer=opt,
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    def train(self, train, test, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((self.batch_size,), dtype=np.int8)
        rewards, dones, values = np.empty((3, self.batch_size))
        observations = np.empty((self.batch_size,) + self.dqn_para['img_size'])

        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]

        # next_obs = env.reset()
        for update in range(updates):
            #  TODO - alll samples
            print(len(train))
            step = 0
            for i, img in enumerate(train):
                img = train[i]
                state = img['data']

                observations[step] = state
                # TODO - Normalisation
                actions[step], values[step] = self.model.action_value(state[None, :])

                next_state = train[(i + 1) % len(train)]['data']

                if actions[step] > len(self.baskets.keys()) - 1:
                    actions[step] = 0


                # try:
                #     print(actions[step])
                #     print('WDNMD1')
                # except:
                #     print('EXCEPTION!!!')
                #     pass
                #
                # try:
                #     print(list(self.baskets.keys())[actions[step]])
                #     print('WDNMD2')
                # except:
                #     print(actions[step])
                #     print(self.baskets)
                #     print('FUCK EXCEPTION!!!')
                #     pass

                basket_key = list(self.baskets.keys())[actions[step]]
                correct_label = img['label']
                reward = 1 if basket_key in correct_label else -1

                rewards[step], dones[step] = reward, None

                # next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]

                # train a batch
                if step >= self.batch_size - 1 or i == len(train) - 1:
                    _, next_value = self.model.action_value(next_state[None, :])
                    returns, advs = self._returns_advantages(rewards, dones, values, next_value)

                    # a trick to input actions and advantages through same API
                    acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
                    # performs a full training step on the collected batch
                    # note: no need to mess around with gradients, Keras API handles it
                    losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
                    print("[%d/%d] Losses: %s" % (update + 1, updates, losses))
                    step = 0
                else:
                    step += 1
        return ep_rews

    def test(self, data):
        rewards = []

        for i, img in enumerate(data):
            state = img['data']
            action, value = self.model.action_value(state[None])
            if action > len(self.baskets.keys()) - 1:
                action = 0
            basket_key = list(self.baskets.keys())[action]
            correct_label = img['label']
            reward = 1 if basket_key in correct_label else -1
            print(basket_key)
            rewards.append(reward)

        return np.array(rewards)

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.value * losses.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = losses.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = losses.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.entropy * entropy_loss


# if __name__ == '__main__':
#     logging.getLogger().setLevel(logging.INFO)
#
#     env = gym.make('CartPole-v0')
#     model = Model(num_actions=env.action_space.n)
#     agent = A2CAgent(model)
#
#     rewards_history = agent.train(env)
#     print("Finished training.")
#     print("Total Episode Reward: %d out of 200" % agent.test(env, True))
#
#     plt.style.use('seaborn')
#     plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.show()