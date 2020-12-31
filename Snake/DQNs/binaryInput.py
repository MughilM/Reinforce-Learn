"""
File: binaryInput.py
Location: /Snake/DQNs
Creation Date: 2020-12-30

This file implements a deep Q-learning network
based on the binary input that is shown in the Q-table
approach. Like that one, we will have 11 binary inputs
corresponding to the state, and it will output one of
3 directions, forward, left, or right. To optimize
learning, I'll use the Double Deep Q-learning
approach, with a target network and a prediction network.
"""

import numpy as np
from ..SnakeAgent import SnakeAgent
from ..SnakeEnv import SnakeGame
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# The methods for producing proper states
# have already been done for us. For starters,
# we just need to need to play out the game


class BinaryDQN:
    def __init__(self, episodes=10, memoryLength=5000, replaceFrequency=3, boardSize=10):
        self.episodes = episodes  # How many times to gather experiential memory?
        self.memoryLength = memoryLength  # The number of actions to store in the memory buffer
        self.replaceFrequency = replaceFrequency  # After how many episodes do we replace the prediction with target?

        self.env = SnakeGame(boardSize=boardSize)
        self.agent = SnakeAgent(environment=self.env)

        self.targetModel = self.createMethod()  # The model which is trained
        self.predictionModel = self.createMethod()  # The model which only gives us Q-value predictions...

    def createMethod(self):
        model = Sequential()
        model.add(Input(shape=(11,), name='Snake Input', dtype=int))
        model.add(Dense(8, activation='relu', name='Hidden Layer 1'))
        model.add(Dense(8, activation='relu', name='Hidden Layer 2'))
        model.add(Dense(8, activation='relu', name='Hidden Layer 3'))
        # The output layer is size 3 (F, L, R), and does not have
        # any specific activation, since it's the Q-value...
        model.add(Dense(3, name='Q Value Output'))
        return model


