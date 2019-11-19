import numpy as np
import random

# Initialise Q-table
q = np.zeros((4, 5))

# Create R-table
# state|action put to another || continue || put to specific || ask || stop
# agree        -10            || 10       || -10             || -10 || -20
# disagree     10             || -10      || 5               || 15  || -20
# correct      -5             || -15      || 20              || -5  || -20
# finish       -10            || -10      || -10             || -10 || 20

r = np.array([[-10, 10, -10, -10, -20], [10, -10, 5, 15, -20], [-5, -15, 20, -5, -20], [-10, -10, -10, -10, -20]])

# Gamma
gamma = 0.8

# Start training
for i in range(10000):
    # Random intialised position
    state = random.randint(0, 3)

    while state != 3:
        # randomly choose action given by human
        next_state = random.randint(0, 3) # TODO
        q[state, next_state] = r[state, next_state] + gamma * q[next_state].max()
        state = next_state

print(q)

# # Test
# for i in range(10):
#     print('test ' + str(i) + ':')
#
#     state = random.randint(0, 5)
#     print('robot in ' + str(state))
#     count = 0
#     while state != 5:
#         if count > 20:
#             print('failed')
#             break
#
#         # choose max q
#         q_max = q[state].max()
#
#         q_max_action = []
#         for action in range(6):
#             if q[state, action] == q_max:
#                 q_max_action.append(action)
#
#         next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]
#         print('robot go to ' + str(next_state))
#         state = next_state
#         count += 1




