import random
from models.TDModel import TDModel


class QLearningModel(TDModel):
    def learn(self, state, action, reward, next_state):
        current_q = self.q[state][action]

        if next_state:
            # Bellman function
            new_q = reward + self.gamma * max(self.q[next_state])
            self.q[state][action] += self.alpha * (new_q - current_q)
        else:
            self.q[state, action] += self.alpha * (reward - current_q)
