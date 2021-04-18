"""
File: experienceReplay.py
Creation Date: 2021-04-15
Location: rl/implementation/base/utils

This file contains the basic class definition of experience replay.
We use a deque to handle automatic pushing out of the old experience whenever we
append a new state...
"""
import numpy as np
from collections import deque


class ExperienceReplay:
    def __init__(self, maxSize):
        self.expSize = maxSize
        self.experience = deque(maxlen=maxSize)

    def __len__(self):
        return len(self.experience)

    def add(self, stateTuple):
        self.experience.append(stateTuple)

    def sample(self, batchSize):
        # Choose without replacement the indices
        indices = np.random.choice(len(self.experience), size=batchSize, replace=False)
        states, actions, rewards, nextStates, dones = zip([self.experience[index] for index in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(nextStates), np.array(dones)
