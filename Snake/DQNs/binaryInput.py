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
from collections import deque

# The methods for producing proper states
# have already been done for us. For starters,
# we just need to need to play out the game


class BinaryDQN:
    def __init__(self, episodes=2500, memoryLength=250, replaceFrequency=100, batchSize=32, boardSize=10):
        self.episodes = episodes  # How many times to gather experiential memory?
        self.episodeCount = 1
        self.memoryLength = memoryLength  # The number of actions to store in the memory buffer
        self.replaceFrequency = replaceFrequency  # After how many episodes do we replace the prediction with target?

        self.agent = SnakeAgent()
        self.env = SnakeGame(boardSize=boardSize, snakeAgent=self.agent)

        self.targetModel = self.createMethod()  # The model which is trained
        self.predictionModel = self.createMethod()  # The model which only gives us Q-value predictions...

        print(f'Model Summary\n{self.targetModel.summary()}')

        # Epsilon, epsilon decay rate, and minimum epsilon
        self.epsilon = 1
        self.minEpsilon = 0.05
        self.epsilonDecayFactor = 0.997

        # The memory is a deque data structure, where if we add something that
        # exceeds the max length, the oldest element simply gets returned...
        self.memory = deque(maxlen=self.memoryLength)
        self.batchSize = batchSize

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

    def preprocessState(self, state):
        """
        The state returned from our snake is a 11-digit bit
        string. We need to convert it to an actual integer
        array of 0s and 1s.
        :return: The preprocessed state to be fed into a
        model.
        """
        return np.asarray(map(int, state))[np.newaxis, :]

    def addExperienceMemory(self):
        """
        Using the given memory length, we fill up a memory buffer
        with state, action, reward, and next state tuples. The goal is
        for these to be passed into the model for training...
        If a game over is encountered during the fill-up,
        then the environment and agent are reset, and the game
        starts again...
        :return: A memory list of state, action, reward, next state,
        and game over tuples.
        """
        # We add batch size number of states. At the start, the memory
        # won't be that full, but it's fine, because it's only a couple
        # of rounds...We keep track of what episode we're on...
        for _ in range(self.batchSize):
            currState = self.env.encodeCurrentState()
            # We do an epsilon greedy action selection,
            if np.random.rand() < self.epsilon:
                actionIndex = np.random.choice(self.agent.actionList)
            else:  # Otherwise, we pass the preprocessed state through the network and choose the best one...
                actionOutput = self.predictionModel.predict(self.preprocessState(currState))
                actionIndex = np.argmax(actionOutput[0])
            chosenAction = self.agent.actionList[actionIndex]
            # Step forward...
            _, reward, gameOver = self.env.stepForward(chosenAction)
            # Encode the new state...
            nextState = self.env.encodeCurrentState()
            # Add the tuple to the memory buffer. If it was a
            # game over, then increment the episode count...
            self.memory.append([currState, chosenAction, reward, nextState, gameOver])
            if gameOver:
                self.env.reset()
                self.episodeCount += 1








