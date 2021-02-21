"""
File: binaryInput.py
Location: /snakeOld/DQNs
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
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from collections import deque
import pprint

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
        print('Compiling models...')
        self.targetModel.compile(optimizer='adam', loss='mse')
        self.targetModel.compile(optimizer='adam', loss='mse')

        # Epsilon, epsilon decay rate, and minimum epsilon
        self.epsilon = 1
        self.minEpsilon = 0.05
        self.epsilonDecayFactor = 0.997
        self.gamma = 0.99
        self.lr = 1e-3

        # The memory is a deque data structure, where if we add something that
        # exceeds the max length, the oldest element simply gets returned...
        self.memory = deque(maxlen=self.memoryLength)
        self.batchSize = batchSize

    def createMethod(self):
        model = Sequential()
        model.add(Input(shape=(11,), name='SnakeInput', dtype=tf.int32))
        model.add(Dense(8, activation='relu', name='HiddenLayer1'))
        model.add(Dense(8, activation='relu', name='HiddenLayer2'))
        model.add(Dense(8, activation='relu', name='HiddenLayer3'))
        # The output layer is size 3 (F, L, R), and does not have
        # any specific activation, since it's the Q-value...
        model.add(Dense(3, name='QValueOutput'))
        return model

    def preprocessState(self, state):
        """
        The state returned from our snake is a 11-digit bit
        string. We need to convert it to an actual integer
        array of 0s and 1s.
        :return: The preprocessed state to be fed into a
        model.
        """
        return np.asarray(list(map(int, state)))[np.newaxis, :]

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
                actionIndex = np.random.choice(np.arange(len(self.agent.actionList)))
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
            self.memory.append([currState, actionIndex, reward, nextState, gameOver])
            # If the game is over, reset the environment, increment the
            # episode count, and decay the epsilon.
            if gameOver:
                self.env.reset()
                self.episodeCount += 1
                self.epsilon *= self.epsilonDecayFactor

    def sampleExperienceReplay(self):
        """
        Using the batch size, returns a random sample of the replay. Additionally,
        it unpacks the states, actions, and returns. This is for easier feeding into
        the model. It also preprocesses the states...
        :return: A 5-tuple of the states, actions, rewards, next states, and
        game overs, each in numpy format.
        """
        memoryLength = len(self.memory)
        # Choose a random set of indices
        chosenIndices = np.random.choice(np.arange(memoryLength), size=self.batchSize, replace=False)
        sampledData = np.asarray([self.memory[index] for index in chosenIndices], dtype=object)
        states = np.squeeze([self.preprocessState(state) for state in sampledData[:, 0]])
        actions = sampledData[:, 1].astype(int)
        rewards = sampledData[:, 2].astype(int)
        nextStates = np.squeeze([self.preprocessState(state) for state in sampledData[:, 3]])  # Should also be (batchSize, 11)
        gameOvers = sampledData[:, 4].astype(bool)
        return states, actions, rewards, nextStates, gameOvers

    def trainStep(self):
        """
        Trains for just ONE BATCH of memory. If we have reached the number of batches where
        it's time to replace the prediction with the target, it will do that too.
        :return: Nothing...
        """
        self.addExperienceMemory()  # First add some memory...
        # Grab the data...
        states, actions, rewards, nextStates, gameOvers = self.sampleExperienceReplay()
        # Eventually, when we call "train" on our network, the 'y' should be a
        # matrix. However, we only have 'y' values for the actions we took, not
        # the actions we didn't take. It is ideal if we did NOT adjust these weights.
        # To make sure the non-actions' weights don't change, the 'y' values for
        # these actions WILL BE THE SAME as the output of the TARGET network..
        # So start with the outputs from the TARGET network...
        currentQValues = self.targetModel.predict(states)
        # Get the maximum Q values for the next states...
        nextQValues = self.predictionModel.predict(nextStates)
        maxNextQ = np.max(nextQValues, axis=1)
        # On our currentQValues matrix, update the Q-values
        # corresponding to the actions with the update formula...
        qValsToUpdate = currentQValues[np.arange(currentQValues.shape[0]), actions]
        currentQValues[np.arange(currentQValues.shape[0]), actions] = (1 - self.lr) * qValsToUpdate + \
            self.lr * (rewards + self.gamma * maxNextQ)
        # Now our matrix is ready for training...
        self.targetModel.train_on_batch(x=states, y=currentQValues)

if __name__ == '__main__':
    binaryDQN = BinaryDQN(boardSize=15)
    binaryDQN.addExperienceMemory()
    print(f'Experience Memory (Length = {len(binaryDQN.memory)}): {pprint.pformat(binaryDQN.memory)}')
    states, actions, rewards, nextStates, gameOvers = binaryDQN.sampleExperienceReplay()
    print(f'States: {states}')
    print(f'Next States: {nextStates}')
    binaryDQN.trainStep()

