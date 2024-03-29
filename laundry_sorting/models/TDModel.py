import numpy as np
import random
import pprint

class TDModel:
    def __init__(self, nob, colours, types, num):
        # Parameters
        self.alpha = 0.5
        self.gamma = 0.8
        self.epsilon = 0.1

        self.colours = colours
        self.types = types
        self.num = num

        # Reward table
        # There is no actually Reward table, and the reward will be calculated after the action is decided
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

    def extend_q_table(self):
        q_table = self.get_q_table()
        self.set_q_table(
            np.insert(q_table, q_table.shape[1], values=np.zeros((q_table.shape[0], 1)).transpose(), axis=1))

    def train(self, noi, data, baskets, interval):
        # Start training
        acc = []
        xs = []

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
                # TODO - Next state 不应该是i_id + 1
                next_state = None
                if i_id + 1 in clothes:
                    next_cloth = clothes[i_id + 1]
                    next_state = self.get_state(next_cloth['i_colour'], next_cloth['i_type'])

                # Learn
                self.learn(state, action, reward, next_state)

            if i > 0 and i % interval == 0:
                acc.append(self.test(data, baskets))
                xs.append(i)

        return [acc, xs]

    def train_with_single_action(self, nop, cloth, baskets, reward_scale):
        # Start training
        for i in range(nop):
            # Get the state
            state = self.get_state(cloth['i_colour'], cloth['i_type'])

            # Using epsilon policy to select action
            action = self.get_action(state)

            # Get the reward
            reward = self.get_reward(label=list(baskets.keys())[action], cloth=cloth) * reward_scale

            next_state = state
            self.learn(state, action, reward, next_state)

    def test(self, data, baskets):
        total_accuracy = 0
        for p_id in range(1, self.num + 1):
            clothes = data[p_id]
            results = {}
            for i_id in clothes.keys():
                cloth = clothes[i_id]
                correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]

                i_colour = cloth['i_colour']
                i_type = cloth['i_type']
                colour_index = self.colours.index(i_colour)
                type_index = self.types.index(i_type)
                state = colour_index * len(self.types) + type_index

                actions = self.q[state]
                action = np.argmax(actions)
                label = list(baskets)[action]

                result = 1 if label in correct_label else 0
                results[i_id] = result

            total_accuracy += (sum(results.values()) / len(results)) / 30
        return total_accuracy

class QLearningModel(TDModel):
    def learn(self, state, action, reward, next_state):
        current_q = self.q[state][action]

        if next_state:
            # Bellman function
            new_q = reward + self.gamma * max(self.q[next_state])
            self.q[state][action] += self.alpha * (new_q - current_q)
        else:
            self.q[state, action] += self.alpha * (reward - current_q)


class SarsaModel(TDModel):
    def learn(self, state, action, reward, next_state):
        current_q = self.q[state][action]

        if next_state:
            # Bellman function
            next_action = self.get_action(next_state)
            new_q = reward + self.gamma * self.q[next_state][next_action]
            self.q[state][action] += self.alpha * (new_q - current_q)
        else:
            self.q[state, action] += self.alpha * (reward - current_q)
