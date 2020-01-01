import numpy as np
import random

class QLearning:
    def __init__(self):
        # Initialise Q-table
        self.q = np.zeros((3, 3))

        # Create Reward table
        # state\action   | choose another || continue || ask for guide ||
        # correct        | -1             || 1        || -1            ||
        # incorrect      | 1              || -1       || 1             ||
        # stop           | 0              || 0        || 0             ||

        self.r = np.array([[-1, 1, -1], [1, -1, 1], [0, 0, 0]])

        # Parameters
        self.gamma = 0.8

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

                # Randomly a label for current cloth
                random_label = random.sample(baskets.keys(), 1)
                # label_name = baskets[random_label]

                # Put cloth into the basket
                robot.pick(i_id)
                robot.moving(random_label[0])
                robot.put(random_label[0])

                # Check result
                correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]
                state = 0 if random_label in correct_label else 1

                actions = []
                for action in range(3):
                    if self.r[state, action] >= 0:
                        actions.append(action)

                # Choose next label for next cloth
                next_state = 2
                if i_id < 16:
                    next_label = random.sample(baskets.keys(), 1)
                    next_cloth = clothes[i_id + 1]
                    next_correct_label = [next_cloth['bc_id_1'], next_cloth['bc_id_2']]
                    next_state = 0 if next_label in next_correct_label else 1

                self.q[state, next_state] = self.r[state, next_state] + self.gamma * self.q[next_state].max()
                state = next_state

    def get_result(self):
        return self.q