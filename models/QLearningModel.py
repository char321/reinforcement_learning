import numpy as np
import random

class QLearningModel:
    def __init__(self):

        # Create Reward table
        #                   ‘white’           ‘dark’      ‘colours’
        #        action  || label 1        || label 2  || label 3       || add new label
        #  state
        # white          || 1              || -1       || -1            ||
        # black          || -1             || 1        || 1             ||
        # dark colour    || -1             || 1        || -1            ||
        # bright colour  || 0              || 0        || 0             ||

        # TODO
        # t-shirt        || 0              || 1        || -1            ||
        # polo           || 0              || 1        || -1            ||
        # pants          || 0              || 1        || -1            ||
        # jeans          || 0              || 1        || -1            ||
        # shirt          || 0              || 1        || -1            ||
        # socks          || 0              || 1        || -1            ||
        # skirt          || 0              || 1        || -1            ||
        # others         || 0              || 1        || -1            ||

        # special material

        self.r = np.array([[-1, 1, -1], [1, -1, 1], [0, 0, 0]])

        # Parameters
        self.gamma = 0.8

        self.colours = ['white', 'black', 'dark', 'colours']

        self.types = ['t-shirt', 'socks', 'polo', 'pants', 'jeans', 'shirt', 'skirt', 'others']

        # Initialise Q-table
        self.q = np.zeros((len(self.colours) * len(self.types), 3))

    def get_reward(self, label, correct_label):
        if label in correct_label:
            return 1
        else:
            return -1

    def train(self, gamma, noi, data, baskets, robot):
        self.gamma = gamma
        # Start training
        for i in range(noi):
            # Choose a random people
            p_id = random.randint(1, 30)

            clothes = data[p_id]

            # Train with an entire sorting procedure
            for i_id in clothes.keys():
                cloth = clothes[i_id]

                # Get the information of cloth
                i_colour = robot.map_to_colour(cloth['i_colour'])
                i_type = robot.map_to_type(cloth['i_type'])

                # Check result
                correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]

                # Use information of cloth as state
                colour_index = self.colours.index(i_colour)
                type_index = self.types.index(i_type)
                state = colour_index * len(self.types) + type_index

                actions = []
                rewards = []
                for action in range(3):
                    label = list(baskets.values())[action]
                    reward = self.get_reward(label, correct_label)
                    actions.append(action)
                    rewards.append(reward)

                # Randomly choose an action that max the reward
                max_reward = max(rewards)
                random_action = random.sample(actions, 1)
                print(random_action)
                print(rewards)
                while rewards[random_action[0]] != max_reward:
                    random_action = random.sample(actions, 1)
                random_label = list(baskets.values())[random_action[0]]
                # label_name = baskets[random_label]

                # Put cloth into the basket
                robot.pick(i_id, i_colour, i_type)
                robot.moving(random_label)
                robot.put(random_label)

                # Choose next label for next cloth
                next_state = 'stop'
                if i_id < 16:
                    next_cloth = clothes[i_id + 1]
                    next_colour_index = self.colours.index(robot.map_to_colour(next_cloth['i_colour']))
                    next_type_index = self.types.index(robot.map_to_type(next_cloth['i_type']))
                    next_state = next_colour_index * len(self.types) + next_type_index
                    self.q[state, random_action] = max_reward + self.gamma * self.q[next_state].max()
                else:
                    self.q[state, random_action] = max_reward + 0
                state = next_state

    def get_result(self):
        return self.q