import random
from models.TDModel import TDModel


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
