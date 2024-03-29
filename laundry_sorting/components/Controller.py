import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import random
from data_loader.DataLoader import DataLoader
from models.TDModel import QLearningModel
from models.TDModel import SarsaModel
from models.DQN import Model
from models.DQN import DQNAgent
from models.A2C import A2CAgent
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

# from google.colab import drive
# drive.mount('/content/drive')
# root_path = 'drive/My Drive/pre_trained_test1'
# root_path = '.'

random.seed(3)

class Controller:
    def __init__(self, config):
        self.robot = Robot()
        self.dataloader = DataLoader()
        self.data = self.dataloader.load_all_data()
        self.baskets = config.baskets.copy()
        self.img_dict = config.img_dict
        self.nob = len(self.baskets)  # number of baskets
        self.mob = 6  # max number of baskets
        self.config = config
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/results/'
        self.set_model()
        self.user = None
        self.default_policy = None
        np.random.seed(100)

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
        train_results = {}
        test_results = {}
        results = {}
        random.seed(p_id)
        train = random.sample(clothes.keys(), int(len(clothes) * 0.7))
        for i_id in clothes.keys():
            cloth = clothes[i_id]
            correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]

            label = self.assign_label(q_table, cloth)
            # print(label)
            # print(correct_label)
            result = 1 if label in correct_label else 0
            if i_id in train:
                train_results[i_id] = result
            else:
                test_results[i_id] = result

            results[i_id] = result
        print("Person %s" % str(p_id))
        print(train_results)
        print(test_results)

        return [train_results, test_results]

    def test_all(self):
        total_accuracy = 0
        for p_id in range(1, 31):
            results = self.test_person(p_id)
            total_accuracy += (sum(results.values()) / len(results)) / 30
        print(total_accuracy)

    def apply(self, p_id):
        self.model.set_parameters(self.config.update_alpha, self.config.update_gamma, self.config.update_epsilon)
        # print('Applying...')

        train_acc = []
        test_acc = []
        xs = []
        count = 1
        clothes = self.data[p_id]
        random.seed(p_id)
        train = random.sample(clothes.keys(), int(len(clothes) * 0.7))
        for i_id in clothes:
            if i_id not in train:
                print(i_id)
                # pass
                continue
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
            [train_result, test_result] = self.test_person(p_id)
            print('WDNMD')
            train_acc.append(sum(train_result.values()) / len(train_result))
            test_acc.append(sum(test_result.values()) / len(test_result))
            xs.append(count)
            count += 1

        title = self.config.model + ' on Person ' + str(p_id)
        name = self.config.model + '-' + str(p_id)

        # plot
        # self.plot_image2(train_acc, test_acc, xs, int(len(clothes) * 0.7), title, name)

        # print(self.baskets)
        # print(self.get_q_table()[0])
        # print(q_table)

    def plot_image2(self, ys_train, ys_test, xs, x_axis, title, name):
        plt.figure()
        plt.axis([1, x_axis, 0, 1])
        plt.plot(xs, ys_train)
        plt.plot(xs, ys_test)
        plt.legend(('Train', 'Test'))
        plt.xticks(range(1, x_axis, int(x_axis / 10)))
        plt.yticks(np.arange(0, 1.2, 0.1))
        plt.xlabel('Iteration Time')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.savefig(self.path + name + '.png')
        plt.close()

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
        dqn_para = self.config.dqn_para.copy()
        episode = dqn_para['episode']

        # agent = DQNAgent(model, target_model, lr, gamma, epsilon, batch_size, buffer_size, self.baskets, img_size,
        #                  target_update_iter, start_learning)

        agent = DQNAgent(dqn_para, self.baskets)

        self.images_data = self.dataloader.image_aug(isCommon=True)

        all_images = []
        for p_id in range(1, 31):
            images = self.images_data[p_id]
            np.random.shuffle(images)
            all_images.extend(images)

        np.random.shuffle(all_images)

        # train, all_images = train_test_split(all_images, test_size=0.002)
        train, test = train_test_split(all_images, test_size=0.3)
        print(np.shape(train))
        agent.evaluation(test)

        print('Parameters')
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(dqn_para)

        print('Start train...')
        agent.train(train, test, episode)

        # agent.model.save_weights('./checkpoints/my_checkpoint')
        # # agent.model.save('my_model.h5')
        #
        # newmodel = Model(3)
        # newmodel.load_weights('./checkpoints/my_checkpoint')
        #
        # neweagent = DQNAgent(dqn_para, self.baskets)
        # neweagent.model = newmodel
        # neweagent.target_model = newmodel
        #
        # print(agent.evaluation(train))
        # print(neweagent.evaluation(train))
        # print(agent.evaluation(test))
        # print(neweagent.evaluation(test))

    def apply_with_dqn(self, p_id):
        self.set_user(p_id)
        self.images_data = self.dataloader.imgaug_data if self.dataloader.imgaug_data else self.dataloader.image_aug()
        all_images = []
        images = self.images_data[p_id]
        random.seed(p_id)
        np.random.shuffle(images)
        all_images.extend(images)
        np.random.shuffle(all_images)

        # train, all_images = train_test_split(all_images, test_size=0.05)
        train, test = train_test_split(all_images, test_size=0.3)
        print(np.shape(train))

        dqn_para = self.config.apply_dqn_para.copy()
        self.baskets = self.config.baskets.copy()
        agent = DQNAgent(dqn_para, self.baskets)
        agent.load_model()

        print('Parameters')
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(dqn_para)

        print('Start apply...' + str(p_id))
        best_train_acc = 0
        best_test_acc = 0

        if_max = False
        previous_correct, previous_wrong = 0, 0
        for i_episode in range(1, dqn_para['episode'] + 1):
            loss = None
            batch_num = 0
            for i, img in enumerate(train):
                state = img['data']
                state = tf.cast(state, tf.float32)
                state = agent.normalisation(state)  # Normalisation
                best_action, q_values = agent.model.action_value(state[None])
                action = agent.get_action(best_action)  # get the real action

                next_state = train[(i + 1) % len(train)]['data']
                next_state = agent.normalisation(next_state)  # Normalisation
                basket_key = list(agent.baskets.keys())[action]
                correct_label = img['label']
                reward = 1 if basket_key in correct_label else -1
                done = None

                if reward == -1:
                    # simulate to ask label from user
                    asked_label = self.ask_for_label_with_img(img)
                    if asked_label == 0:
                        continue

                    if asked_label not in self.baskets:

                        if self.nob >= self.mob and not if_max:
                            print('Already achieve maximum number of baskets!')
                            if_max = True
                        else:
                            print('Add basket ' + str(asked_label))
                            self.baskets = self.robot.add_new_label(asked_label, self.baskets)
                            self.nob += 1

                            agent.update_action_dim()
                            agent.update_target_model()
                            agent.compile(dqn_para)

                            # # for j in range(np.shape(agent.model.layers)[0]):
                            # print(agent.model.layers[1].get_weights())
                            agent.baskets = self.baskets
                            print(agent.baskets)

                if previous_correct == 0 and previous_wrong == 0:
                    if reward == 1:
                        previous_correct += 1
                    else:
                        previous_wrong += 1
                elif previous_correct > 0 and previous_wrong == 0:
                    if reward == 1:
                        previous_correct = previous_correct + 1 if previous_correct <= 6 else previous_correct
                    else:
                        previous_correct = 0
                        previous_wrong += 1
                elif previous_correct == 0 and previous_wrong > 0:
                    if reward == 1:
                        previous_correct += 1
                        previous_wrong = 0
                    else:
                        previous_wrong = previous_wrong + 1 if previous_wrong <= 6 else previous_wrong

                if reward == 1:
                    if previous_correct > 3 and previous_correct <= 5:
                        reward = 3
                    if previous_correct > 5 and previous_correct <= 6:
                        reward = 5
                else:
                    if previous_wrong > 3 and previous_wrong <= 5:
                        reward = -3
                    if previous_wrong > 5 and previous_wrong <= 6:
                        reward = -5
                agent.store_transition(state, action, reward, next_state, done)
                agent.num_in_buffer = min(agent.num_in_buffer + 1, agent.buffer_size)

                if i > agent.start_learning:  # start learning
                    losses = agent.train_step()
                    loss = losses if not loss else min(loss, losses)
                    # print('Batch ' + str(batch_num) + 'Loss: ' + str(losses))
                    batch_num += 1
                if i % agent.target_update_iter == 0:
                    agent.update_target_model()

                # state = None if done else next_state

            train_rewards = agent.evaluation(train)
            train_acc = sum(train_rewards == 1) / len(train_rewards)
            test_rewards = agent.evaluation(test)
            test_acc = sum(test_rewards == 1) / len(test_rewards)
            print('Episode ' + str(i_episode) + ' Loss: ' + str(loss) + ' Train Accuracy: ' +
                  str(train_acc) + ' Test Accuracy: ' + str(test_acc))
            if train_acc + test_acc > best_train_acc + best_test_acc:
                best_train_acc = train_acc
                best_test_acc = test_acc
                agent.save_model(p_id)

                model_info = dqn_para.copy()
                model_info['baskets'] = str(self.baskets)
                model_info['best_train_acc'] = str(best_train_acc)
                model_info['best_test_acc'] = str(best_test_acc)
                model_info['episode'] = i_episode
                # for s in model_info.keys():
                #     print(s)
                #     print(type(s))
                json_str = json.dumps(model_info)
                with open(root_path + '/checkpoints/' + str(p_id) + '/model_info.json', 'w') as json_file:
                    json_file.write(json_str)

    def train_with_a2c(self):

        dqn_para = self.config.dqn_para.copy()
        episode = dqn_para['episode']

        # agent = DQNAgent(model, target_model, lr, gamma, epsilon, batch_size, buffer_size, self.baskets, img_size,
        #                  target_update_iter, start_learning)

        agent = A2CAgent(dqn_para, self.baskets)

        self.images_data = self.dataloader.image_aug(isCommon=True)

        all_images = []
        for p_id in range(1, 31):
            images = self.images_data[p_id]
            np.random.shuffle(images)
            all_images.extend(images)

        np.random.shuffle(all_images)

        train, all_images = train_test_split(all_images, test_size=0.02)
        train, test = train_test_split(all_images, test_size=0.15)
        print(np.shape(train))

        rewards_history = agent.train(train, test, 100)

        print(rewards_history)

        results = agent.test(test)
        print(results)
