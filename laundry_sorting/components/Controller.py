import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import matplotlib.pyplot as plt
from data_loader.DataLoader import DataLoader
from models.TDModel import QLearningModel
from models.TDModel import SarsaModel
from models.DQN import DQN
from components.User import User
from components.Robot import Robot
from components.Config import Config


# persons
# - key: person id
# - value: clothes sorting information (as a dict)
#   clothes:
#       - key: i_id
#       - value: clothe sorting information including colour, type, basket id, basket category id, basket labe

class Controller:
    def __init__(self, config):
        self.robot = Robot()
        self.dataloader = DataLoader()
        self.data = self.dataloader.load_all_data()
        self.baskets = {1: 'white', 3: 'dark', 5: 'colour'}
        # self.baskets = {1: 'whites', 2: 'lights', 3: 'darks', 4: 'brights', 5: 'colours',
        #                 6: 'handwash', 7: 'denims', 8: 'delicates', 9: 'children', 10: 'mixed', 11: 'miscellaneous'}
        self.nob = len(self.baskets)  # number of baskets
        self.mob = 6  # max number of baskets
        self.config = config
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/results/'
        self.set_model()
        self.user = None
        self.default_policy = None

    def set_model(self):
        if self.config.model == 'QLearning':
            self.model = QLearningModel(self.nob, self.dataloader.get_colours(), self.dataloader.get_types(),
                                        self.config.num)
        if self.config.model == 'Sarsa':
            self.model = SarsaModel(self.nob, self.dataloader.get_colours(), self.dataloader.get_types(),
                                    self.config.num)

    def set_user(self, p_id):
        self.user = User(p_id, self.data[p_id])

    def set_q_table(self, q_table):
        self.model.set_q_table(q_table)

    def get_q_table(self):
        return self.model.get_q_table()

    def ask_for_label(self, cloth):
        return self.user.guide_label(cloth)

    def assign_label(self, q_table, cloth):
        # Get the information of cloth
        i_colour = cloth['i_colour']
        i_type = cloth['i_type']
        colour_index = self.model.colours.index(i_colour)
        type_index = self.model.types.index(i_type)
        state = colour_index * len(self.model.types) + type_index

        actions = q_table[state]
        action = np.argmax(actions)
        label = list(self.baskets.keys())[action]

        return label

    def reload_default_policy(self):
        if not self.default_policy:
            print('The model has not been trained.')
        else:
            self.set_q_table(np.copy(self.default_policy['q']))
            self.nob = self.default_policy['nob']
            self.baskets = dict(self.default_policy['baskets']).copy()

    def train(self):
        self.model.set_parameters(self.config.train_alpha, self.config.train_gamma, self.config.train_epsilon)
        # print('Training...')

        results = self.model.train(self.config.noi, self.data, self.baskets, 10)

        # Store default policy
        self.default_policy = {'q': np.copy(self.get_q_table()), 'nob': self.nob, 'baskets': self.baskets.copy()}

        title = self.config.model + ' with\n' + 'alpha: ' + str(self.config.train_alpha) + ' gamma: ' + str(
            self.config.train_gamma) + ' epsilon: ' + str(self.config.train_epsilon)
        name = self.config.model + '_alpha' + str(self.config.train_alpha) + '_gamma' + str(
            self.config.train_gamma) + '_epsilon' + str(self.config.train_epsilon)

        # plot
        # self.plot_image(results[0], results[1], self.config.noi, title, name)

    def test_person(self, p_id):
        q_table = self.model.get_q_table()
        # print(q_table)

        clothes = self.data[p_id]
        results = {}
        for i_id in clothes.keys():
            cloth = clothes[i_id]
            correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]

            label = self.assign_label(q_table, cloth)
            # print(label)
            # print(correct_label)
            result = 1 if label in correct_label else 0
            results[i_id] = result

        # print("Person %s" % str(p_id))
        # print(results)

        return results

    def test_all(self):
        total_accuracy = 0
        for p_id in range(1, 31):
            results = self.test_person(p_id)
            total_accuracy += (sum(results.values()) / len(results)) / 30
        print(total_accuracy)

    def apply(self, p_id):
        self.model.set_parameters(self.config.update_alpha, self.config.update_gamma, self.config.update_epsilon)
        # print('Applying...')

        acc = []
        xs = []
        count = 1
        clothes = self.data[p_id]
        for i_id in clothes:
            cloth = clothes[i_id]
            q_table = self.model.get_q_table()
            # System make decision
            label = self.assign_label(q_table, cloth)

            # System get the feedback
            response = self.user.get_response(cloth, label)

            if response:  # correct
                # TODO - Any action?
                # print('correct')
                None
            else:  # incorrect
                asked_label = self.ask_for_label(cloth)
                if asked_label == 0:
                    continue

                if asked_label not in self.baskets:
                    # TODO - reference?
                    if self.nob >= self.mob:

                        # print("Already achieve maximum number of baskets!")
                        None
                    else:
                        self.baskets = self.robot.add_new_label(asked_label, self.baskets)
                        self.nob += 1

                        # print("Add new basket: %d" % asked_label)

                        # Extend q_table
                        self.model.extend_q_table()

            # System update the q-table

            # # TODO - DELETE
            state = self.model.get_state(cloth['i_colour'], cloth['i_type'])
            # if state == 0:
            #     print(i_id)
            #     print(response)
            #     print(cloth['i_colour'] + ' - ' + cloth['i_type'])
            #     # print(state)
            #     print(self.get_q_table()[state])
            self.model.train_with_single_action(self.config.nop, cloth, self.baskets, (
                self.config.correct_scale if response else self.config.incorrect_reward))
            # if state == 0:
            #     print(self.get_q_table()[state])

            # TODO
            # test_result = self.test_person(p_id)
            result = self.test_person(p_id)
            acc.append(sum(result.values()) / len(result))
            xs.append(count)
            count += 1

        title = 'Updated ' + str(p_id) + ' using ' + self.config.model + ' with\n' + 'alpha: ' + str(
            self.config.update_alpha) + ' gamma: ' + str(
            self.config.update_gamma) + ' epsilon: ' + str(self.config.update_epsilon)
        name = 'Updated_' + str(p_id) + '_' + self.config.model + '_alpha' + str(
            self.config.update_alpha) + '_gamma' + str(
            self.config.update_gamma) + '_epsilon' + str(self.config.update_epsilon)

        # plot
        # self.plot_image(acc, xs, len(clothes), title, name)

        # print(self.baskets)
        # print(self.get_q_table()[0])
        # print(q_table)

    def plot_image(self, ys, xs, x_axis, title, name):
        plt.figure()
        plt.axis([0, x_axis, 0, 1])
        plt.plot(xs, ys)
        plt.xlabel('Iteration Time')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.savefig(self.path + name + '.png')
        plt.close()

    def train_with_dqn(self):
        images = self.dataloader.load_new_images()

        clothes = self.data[1]
        del(clothes[5])
        del (clothes[6])
        print(clothes)
        with tf.Session() as sess:
            rl = DQN(
                sess=sess,
                s_dim=self.config.state_dim,
                a_dim=len(self.baskets),
                batch_size=1,
                gamma=0.99,
                lr=0.01,
                epsilon=0.1,
                replace_target_iter=5
            )
            tf.global_variables_initializer().run()

            print('here')
            rs = []
            for i_episode in range(100):
                print(i_episode)
                count = 0

                for i_id in list(clothes.keys()): # clothes.keys():

                    cloth = clothes[i_id]
                    state = images[1][i_id]
                    count += 1

                    r_sum = 0
                    # print(state.shape)

                # while True:
                    action = rl.choose_action(state)
                    # print(action)
                    next_state = state
                    if count < len(images) - 2:
                        # print('i_id')
                        # print(i_id)
                        temp_id = list(clothes.keys())[count]
                        # print(temp_id)
                        next_state = images[temp_id][0]

                    basket_key = list(self.baskets.keys())[action]
                    correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]
                    reward = 1 if basket_key in correct_label else -1
                    print(reward)

                    done = False

                    rl.store_transition_and_learn(state, action, reward, next_state, done)

                    r_sum += 1
                    if done:
                        print(i_episode, r_sum)
                        rs.append(r_sum)
                        break

            print('mean', np.mean(rs))
