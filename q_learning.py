import numpy as np
import random

# Initialise Q-table
q = np.zeros((6, 6))

# Create R-table
r = np.array([[-1, -1, -1, -1, 0, -1], [-1, -1, -1, 0, -1, 100], [-1, -1, -1, 0, -1, -1], [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100], [-1, 0, -1, -1, 0, 100]])

# Gamma
gamma = 0.8

# Start training
for i in range(1000):
    # Random intialised position
    state = random.randint(0, 5)

    while state != 5:
        # randomly choose action in R-table with positive reward
        actions = []
        for action in range(6):
            if r[state, action] >= 0:
                actions.append(action)
        next_state = actions[random.randint(0, len(actions) - 1)]
        q[state, next_state] = r[state, next_state] + gamma * q[next_state].max()
        state = next_state

print(q)

# Test
for i in range(10):
    print('test ' + str(i) + ':')

    state = random.randint(0, 5)
    print('robot in ' + str(state))
    count = 0
    while state != 5:
        if count > 20:
            print('failed')
            break

        # choose max q
        q_max = q[state].max()

        q_max_action = []
        for action in range(6):
            if q[state, action] == q_max:
                q_max_action.append(action)

        next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]
        print('robot go to ' + str(next_state))
        state = next_state
        count += 1




