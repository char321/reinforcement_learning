import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from data_loader.DataLoader import DataLoader
from models.TDModel import QLearningModel
from models.TDModel import SarsaModel
from models.DQN import Model
from models.DQN import DQNAgent
from components.User import User
from components.Robot import Robot
from sklearn.model_selection import train_test_split
import tensorflow as tf


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
        self.baskets = config.baskets
        self.img_dict = config.img_dict
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

    def ask_for_label_with_img(self, img):
        return self.user.guide_label_with_img(img)

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
        dqn_para = self.config.dqn_para
        episode = dqn_para['episode']

        # agent = DQNAgent(model, target_model, lr, gamma, epsilon, batch_size, buffer_size, self.baskets, img_size,
        #                  target_update_iter, start_learning)

        agent = DQNAgent(dqn_para, self.baskets)

        self.images_data = self.dataloader.image_aug(isCommon=True)

        # print(len(self.images_data))

        all_images = []
        for p_id in range(1, 31):
            images = self.images_data[p_id]
            np.random.shuffle(images)
            all_images.extend(images)

        np.random.shuffle(all_images)
        #
        # train, all_images = train_test_split(all_images, test_size=0.002)
        train, test = train_test_split(all_images, test_size=0.3)
        print(np.shape(train))
        agent.evaluation(test)
        print('start train')
        agent.train(train, test, episode)

        # agent.model.save_weights('./')

        # TODO
        # save & reload
        # updadte network & save & reload

    def temp(self):
        model = Model(3)
        model.load_weights('./checkpoint')