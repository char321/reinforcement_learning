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
        self.baskets = {
            1: 'white',
            3: 'dark',
            5: 'colour'
        }
        # self.baskets = {1: 'whites', 2: 'lights', 3: 'darks', 4: 'brights', 5: 'colours',
        #                 6: 'handwash', 7: 'denims', 8: 'delicates', 9: 'children', 10: 'mixed', 11: 'miscellaneous'}
        self.img_dict = {
            0: 'og',
            1: 'ud',
            2: 'lr',
            3: 'affine',
            4: 'rot1',
            5: 'rot2',
            6: 'scale',
            7: 'blur',
            8: 'add',
            9: 'com1',
            10: 'com2',
            11: 'com3'
        }
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
        episode = 500
        self.images_data = self.dataloader.image_aug()
        all_images = []
        for p_id in range(1, 31):
            images = self.images_data[p_id]
            all_images.extend(images)

        train, test = train_test_split(all_images, test_size=0.3)
        tf.compat.v1.reset_default_graph()
        with tf.Session() as sess:
            rl = DQN(
                sess=sess,
                s_dim=self.config.state_dim,
                a_dim=len(self.baskets),
                batch_size=int(50 * len(self.img_dict)),
                gamma=0,
                lr=0.00001,
                epsilon=0.1,
                replace_target_iter=int(100 * len(self.img_dict))
            )

            tf.global_variables_initializer().run()

            for i_episode in range(1, episode + 1):

                train_rewards = []
                for i, img in enumerate(train):
                    state = img['data']
                    action = rl.choose_action(state)
                    # print(action)
                    # TODO -random set next state
                    next_state = train[(i + 1) % len(train)]['data']

                    basket_key = list(self.baskets.keys())[action]
                    correct_label = img['label']
                    reward = 1 if basket_key in correct_label else -1
                    # print(reward)
                    train_rewards.append(1 if reward == 1 else 0)

                    done = False
                    rl.store_transition_and_learn(state, action, reward, next_state, done)

                train_accuracy = float(sum(train_rewards)) / float(len(train_rewards))

                test_rewards = []
                for i, img in enumerate(test):
                    state = img['data']
                    action = rl.predict(state)
                    basket_key = list(self.baskets.keys())[action]
                    correct_label = img['label']
                    reward = 1 if basket_key in correct_label else -1
                    # print('Predict item ' + str(img['i_id']) + ' with type ' + str(img['type']) + ': ' + str(reward))
                    test_rewards.append(1 if reward == 1 else 0)
                test_accuracy = float(sum(test_rewards)) / float(len(test_rewards))

                if i_episode % 10 == 0:
                    print('Epsisode ' + str(i_episode) + ' Train Accuracy: ' + str(
                        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

            # save the model
            # tf.train.write_graph(sess.graph_def, './checkpoint_dir', 'model_pre_trained' + '.pbtxt',
            #                      as_text=True)
            #
            # saver = tf.train.Saver()
            # saver.save(sess, './checkpoint_dir/model_pre_trained')

    def apply_with_dqn(self):
        episode = 10
        update_time = 0
        self.images_data = self.dataloader.image_aug()
        images = self.images_data[self.user.get_pid()]
        train, test = train_test_split(images, test_size=0.3)

        best_results = None
        best_accuracy = None
        tf.compat.v1.reset_default_graph()
        with tf.Session() as sess:
            rl = DQN(
                sess=sess,
                s_dim=self.config.state_dim,
                a_dim=len(self.baskets),
                batch_size=int(5 * len(self.img_dict)),
                gamma=0,
                lr=0.00001,
                epsilon=0.1,
                replace_target_iter=int(10 * len(self.img_dict))
            )
            tf.global_variables_initializer().run()

            print('Id: ' + str(self.user.get_pid()))
            for i_episode in range(1, episode + 1):

                train_rewards = []
                for i, img in enumerate(train):
                    state = img['data']
                    action = rl.choose_action(state)
                    # print(action)
                    # TODO -random set next state
                    next_state = train[(i + 1) % len(train)]['data']

                    basket_key = list(self.baskets.keys())[action]
                    correct_label = img['label']
                    reward = 1 if basket_key in correct_label else -1
                    # print(reward)
                    train_rewards.append(1 if reward == 1 else 0)

                    emotion_level = self.user.get_emotion_level(reward == 1)

                    # print('Train with item ' + str(img['i_id']) + ' with type ' + str(img['type']))
                    if reward == -1:
                        # simulate to ask label from user
                        asked_label = self.ask_for_label_with_img(img)
                        if asked_label == 0:
                            continue

                        if asked_label not in self.baskets:

                            if self.nob >= self.mob:
                                print("Already achieve maximum number of baskets!")
                            else:
                                print("Add basket")
                                self.baskets = self.robot.add_new_label(asked_label, self.baskets)
                                self.nob += 1
                                # TODO - update the entire network
                                update_time += 1
                                rl.update_actions()
                                tf.global_variables_initializer().run()
                                rl.copy_net()

                    done = False
                    rl.store_transition_and_learn(state, action, reward, next_state, done)

                train_accuracy = float(sum(train_rewards)) / float(len(train_rewards))
                if not best_accuracy or train_accuracy > best_accuracy:
                    best_results = train_rewards
                    best_accuracy = train_accuracy

                test_rewards = []
                for i, img in enumerate(test):
                    state = img['data']
                    action = rl.predict(state)
                    basket_key = list(self.baskets.keys())[action]
                    correct_label = img['label']
                    reward = 1 if basket_key in correct_label else -1
                    # print('Predict item ' + str(img['i_id']) + ' with type ' + str(img['type']) + ': ' + str(reward))
                    test_rewards.append(1 if reward == 1 else 0)
                test_accuracy = float(sum(test_rewards)) / float(len(test_rewards))

                if i_episode % 10 == 0:
                    print('Epsisode ' + str(i_episode) + ' Train Accuracy: ' + str(
                        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

            # save the model
            tf.train.write_graph(sess.graph_def, './checkpoint_dir', 'model' + str(self.user.get_pid()) + '.pbtxt',
                                 as_text=True)

            saver = tf.train.Saver()
            saver.save(sess, './checkpoint_dir/model' + str(self.user.get_pid()))

            baskets = {str(k): v for k, v in self.baskets.items()}
            model_info = {
                'update_time': update_time,
                'baskets': baskets
            }

            json_str = json.dumps(model_info)
            with open('./checkpoint_dir/model' + str(self.user.get_pid()) + '.json', 'w') as json_file:
                json_file.write(json_str)

            # rewards = []
            # for i, img in enumerate(train):
            #     state = img['data']
            #     action = rl.predict(state)
            #     basket_key = list(self.baskets.keys())[action]
            #     correct_label = img['label']
            #     reward = 1 if basket_key in correct_label else -1
            #     # print('Predict item ' + str(img['i_id']) + ' with type ' + str(img['type']) + ': ' + str(reward))
            #     rewards.append(1 if reward == 1 else 0)
            # print('Accuracy: ' + str(sum(rewards) / float(len(rewards))))

        print('FINISH')
        return (best_results, best_accuracy)

    def test_with_dqn(self):
        p_id = self.user.get_pid()
        self.images_data = self.dataloader.image_aug()
        images = self.images_data[p_id]
        tf.compat.v1.reset_default_graph()

        rewards = []
        with tf.Session() as sess:
            # load model

            saver = tf.train.import_meta_graph('./checkpoint_dir/model' + str(p_id) + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
            # tf.train.write_graph(sess.graph_def, './checkpoint_dir', 'model.pbtxt', as_text=True)

            with open('./checkpoint_dir/model' + str(p_id) + '.json', 'r') as json_file:
                model_info = json.load(json_file)

            self.baskets = {int(k): v for k, v in model_info['baskets'].items()}
            update_time = model_info['update_time']

            rl = DQN(sess=sess, epsilon=0)
            rl.reload_tensor(update_time)

            for i, img in enumerate(images):
                state = img['data']
                action = rl.predict(state)
                basket_key = list(self.baskets.keys())[action]
                correct_label = img['label']
                reward = 1 if basket_key in correct_label else -1
                print('Predict item ' + str(img['i_id']) + ' with type ' + str(img['type']) + ': ' + str(reward))
                rewards.append(1 if reward == 1 else 0)
            print('Accuracy: ' + str(sum(rewards) / float(len(rewards))))

    def temp(self):
        episode = 300
        model = Model(3)
        target_model = Model(3)
        lr = 0.00001
        gamma = 0
        epsilon = 0.1
        batch_size = int(5 * len(self.img_dict))
        buffer_size = int(50 * len(self.img_dict))
        img_size = (400, 300, 3)
        target_update_iter = int(100 * len(self.img_dict))
        start_learning = int(int(10 * len(self.img_dict)))

        # agent = DQNAgent(model, target_model, lr, gamma, epsilon, batch_size, buffer_size, self.baskets, img_size,
        #                  target_update_iter, start_learning)

        agent = DQNAgent(model, target_model, lr, gamma, epsilon, 5, 100, self.baskets, (400, 300, 3),
                         20, 10)

        self.images_data = self.dataloader.image_aug()
        all_images = []
        for p_id in range(1, 31):
            images = self.images_data[p_id]
            all_images.extend(images)

        train, test = train_test_split(all_images, test_size=0.1)
        train, test = train_test_split(test, test_size=0.7)
        agent.evalation(test)
        agent.train(train, test, episode)
