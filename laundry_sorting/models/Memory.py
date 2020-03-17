import collections
import random
import numpy as np


class Memory:
    def __init__(self, batch_size, max_size):
        self.batch_size = batch_size  # size of mini-batch
        self.max_size = max_size  # maximum size of memory
        self.transitions = collections.deque()

    def store_transition(self, s, a, r, s_, done):
        if len(self.transitions) == self.max_size:
            self.transitions.popleft()

        self.transitions.append((s, a, r, s_, done))

    def get_mini_batches(self):
        n_sample = min(self.batch_size, len(self.transitions))
        t = random.sample(self.transitions, k=n_sample)
        t = list(zip(*t))

        return tuple(np.array(e) for e in t)

    def increase_action_dim(self):
        for i in range(len(self.transitions)):
            s, a, r, s_, done = self.transitions.popleft()
            a = np.append(a, 0)
            self.transitions.append((s, a, r, s_, done))
