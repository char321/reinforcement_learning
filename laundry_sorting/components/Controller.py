import pandas as pd
import numpy as np
from data_loader.DataLoader import DataLoader
from models.QLearningModel import QLearningModel
from models.SarsaModel import SarsaModel
from components.User import User
from components.Robot import Robot


# persons
# - key: person id
# - value: clothes sorting information (as a dict)
#   clothes:
#       - key: i_id
#       - value: clothe sorting information including colour, type, basket id, basket category id, basket labe

class Controller:
    def __init__(self):
        self.robot = Robot()
        self.dataloader = DataLoader()
        self.data = self.dataloader.load_sorting_data()
        self.baskets = {1: 'white', 3: 'dark', 5: 'colour'}
        # self.baskets = {1: 'whites', 2: 'lights', 3: 'darks', 4: 'brights', 5: 'colours',
        #                 6: 'handwash', 7: 'denims', 8: 'delicates', 9: 'children', 10: 'mixed', 11: 'miscellaneous'}
        self.nob = len(self.baskets)  # number of baskets
        self.mob = 6  # max number of baskets
        self.model = QLearningModel(self.nob)
        self.user = None

    def set_model(self, model):
        if model == 'QLearning':
            self.model = QLearningModel(self.nob)
        if model == 'Sarsa':
            self.model = SarsaModel(self.nob)

    def set_user(self, p_id):
        self.user = User(p_id, self.data[p_id])

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

    def train(self):
        self.model.set_parameters(alpha=0.5, gamma=0.8, epsilon=0.1)
        noi = 10000  # number of iterations
        print('Training...')

        self.model.train(noi, self.data, self.baskets)

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

        print("Person %s" % str(p_id))
        print(results)
        return results

    def test_all(self):
        total_accuracy = 0
        for p_id in range(1, 31):
            results = self.test_person(p_id)
            total_accuracy += (sum(results.values()) / 16) / 30

        print(total_accuracy)

    def apply(self, p_id):
        self.model.set_parameters(alpha=0.5, gamma=0.8, epsilon=0.1)

        nop = 500  # number of repeat time for each sorting behaviour
        print('Applying...')

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
                print('correct')
            else:  # incorrect
                asked_label = self.ask_for_label(cloth)
                if asked_label not in self.baskets:
                    # TODO - reference?
                    if self.nob >= self.mob:
                        print("Already achieve maximum number of baskets!")
                    else:
                        self.baskets = self.robot.add_new_label(asked_label, self.baskets)
                        self.nob += 1
                        print("Add new basket: %d" % asked_label)

                        # Extend q_table
                        q_table = self.model.get_q_table()
                        q_table = np.insert(q_table, q_table.shape[1],
                                            values=np.zeros((q_table.shape[0], 1)).transpose(),
                                            axis=1)
                        self.model.set_q_table(q_table)

            # System update the q-table
            self.model.train_with_single_action(nop, cloth, self.baskets)
