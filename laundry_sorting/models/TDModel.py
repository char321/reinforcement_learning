import numpy as np
import random

class TDModel:
    def __init__(self, nob):

        # Reward table
        # There is no actually Reward table, and the reward will be calculated after the action is decided

        # Parameters
        self.alpha = 0.5
        self.gamma = 0.8
        self.epsilon = 0.1

        self.colours = ['white', 'black', 'dark', 'colours']

        self.types = ['t-shirt', 'socks', 'polo', 'pants', 'jeans', 'shirt', 'skirt', 'others']

        # Initialise Q-table
        # The number of states = number of colours * number of types
        self.q = np.zeros((len(self.colours) * len(self.types), nob))

    def set_parameters(self, alpha, gamma, epsilon):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon

    def get_reward(self, label, cloth):
        correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]
        return 1 if label in correct_label else -1

    def get_state(self, i_colour, i_type):
        colour_index = self.colours.index(i_colour)
        type_index = self.types.index(i_type)

        return colour_index * len(self.types) + type_index

    def get_action(self, state):
        state_action = self.q[state]

        if np.random.rand() < self.epsilon:
            action = np.random.choice(range(len(state_action)))
        else:
            max_value = max(state_action)
            action = random.choice(range(len(state_action)))
            while state_action[action] != max_value:
                action = random.choice(range(len(state_action)))
        return action

    def get_q_table(self):
        return self.q

    def set_q_table(self, q_table):
        self.q = q_table

    def train(self, noi, data, baskets):
        # Start training
        for i in range(noi):
            # Choose a random people
            p_id = random.randint(1, 30)

            clothes = data[p_id]

            # Train with an entire sorting procedure
            for i_id in clothes.keys():
                cloth = clothes[i_id]

                # Get the state
                state = self.get_state(cloth['i_colour'], cloth['i_type'])

                # Using epsilon policy to select action
                action = self.get_action(state)

                # Get the reward
                reward = self.get_reward(label=list(baskets.keys())[action], cloth=cloth)

                # TODO Put cloth into the basket
                # robot.pick(i_id, i_colour, i_type)
                # robot.moving(random_label)
                # robot.put(random_label)

                # Choose next label for next cloth
                next_state = None
                if i_id < 16:
                    next_cloth = clothes[i_id + 1]
                    next_state = self.get_state(next_cloth['i_colour'], next_cloth['i_type'])

                # Learn
                self.learn(state, action, reward, next_state)

    def train_with_single_action(self, nop, cloth, baskets):
        # Start training
        for i in range(nop):
            # Get the information of cloth
            i_colour = cloth['i_colour']
            i_type = cloth['i_type']

            # Check result
            first_label = cloth['bc_id_1']
            correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]

            # Get the state
            state = self.get_state(cloth['i_colour'], cloth['i_type'])

            # Using epsilon policy to select action
            action = self.get_action(state)

            # Get the reward
            reward = self.get_reward(label=list(baskets.keys())[action], cloth=cloth) * 2

            # TODO Put cloth into the basket
            # robot.pick(i_id, i_colour, i_type)
            # robot.moving(random_label)
            # robot.put(random_label)

            next_state = state
            self.learn(state, action, reward, next_state)

