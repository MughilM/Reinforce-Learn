"""
File: ReplaceDQN.py
Creation Date: 2021-04-01
Location: implementation/base/DQN

This file implements a type of deep Q network used for reinforcement learning training.
I've called Replace DQN, where there are actually two copies of the same network.
One is actually trained (called the target network), and one is used for predictions
of the Q values of the next state (called the prediction network). Every so often,
the weights of the prediction network get overwritten by the weights of the target network.
This is so training is more stable, since training for Q values constitutes aiming at a moving target.
Of course, the network has to be defined separately.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Dict, List
from ..agents.basicAgent import Agent
from ..environment import Environment
from ..utils.experienceReplay import ExperienceReplay
from .BaseDQN import BaseDQN

import numpy as np
import os
import shutil
import sys


class ReplaceDQN(BaseDQN):
    def __init__(self, outputDir, experimentName, agents: Dict[str, Agent], environment: Environment,
                 agentPlayOrder: List[str], updateFreq=10, **kwargs):
        """
        Initialization for a ReplaceDQN, which contains both target and prediction networks.
        The same required variables of a BaseDQN apply, with the addition of an update frequency,
        which says how often to replace the weights of the prediction network with those
        of the target network.
        :param outputDir: The directory which holds all experiment results
        :param experimentName: The name of the experiment (subdirectory created)
        :param agents: Dictionary of agents
        :param environment: The Environment object
        :param agentPlayOrder: The play order of the agents. Who goes first?
        :param updateFreq: After how many train steps before weight replacement in prediction network?
        :param kwargs: Extra arguments for BaseDQN (stateLimit, epsilon, etc.)
        """
        self.updateFreq = updateFreq
        # Create a target network and a prediction network.
        # Technically we could have exactly the same weights, but they're
        # random right now...
        self.targetModel = self.createModel()
        self.predictionModel = self.createModel()
        super().__init__(outputDir, experimentName, agents, environment, agentPlayOrder, **kwargs)

    def createModel(self) -> Model:
        """
        The method used to create the model for target and prediction.
        :return: A TensorFlow Model object.
        """
        raise NotImplementedError('Please implement the createModel function so we can train :)')

    def chooseActionFromPrediction(self, QValPreds):
        """
        Insert documentation here...
        :param QValPreds: The O-value predictions outputted by model.
        :return: The corresponding action to take, after whatever logic.
        """
        raise NotImplementedError('Please implement a way to choose the action based on the predictions.')

    def loadDataArtifacts(self, **kwargs):
        """
        Standard loading of data artifacts from experiment directory
        :param kwargs:
        :return:
        """
        raise NotImplementedError('Please implement loading of custom artifacts (allowed to pass if there are none.)')

    def saveDataArtifacts(self, **kwargs):
        """
        Standard saving of custom data artifacts e.g. plots, csv's.
        :param kwargs:
        :return:
        """
        raise NotImplementedError('Please implement saving of custom artifacts (allowed to pass if there are none.)')

    def train(self, batchSize, trainSteps, resetEnv=False):
        """
        For the ReplaceDQN, the one important change is that we will be using the prediction network
        to make predictions on the Q-values of the next states, instead of the target network.
        Additionally, after a set number of training iterations, we replace the weights in
        the prediction network with those in the target network. This is to improve stability.
        :param batchSize: The training batch size
        :param trainSteps: Number of training iterations to go for
        :param resetEnv: Whether to reset the environment and agents when starting a training loop.
        :return:
        """
        if resetEnv:
            _ = self.env.reset()
            self.turnIndex = 0
        self.agentToPlay = self.agentPlayOrder[self.turnIndex]
        # Loop until you go through the training steps
        for _ in range(trainSteps):
            currentEncodedState = self.env.encodeCurrentState()
            # Take the next action based on the epsilon-greedy approach...
            if np.random.rand() < self.epsilon:
                action = self.env.agents[self.agentToPlay].chooseRandomAction()
            else:
                # We have to push the state through the prediction network...
                prediction = self.predictionModel(np.expand_dims(currentEncodedState, axis=0))
                # Now choose the next action based on whatever logic.
                action = self.chooseActionFromPrediction(prediction)
            # Step forward in the environment based on the action
            nextRawState, reward, gameOver = self.env.stepForward(self.agentToPlay, action)
            # Encode the next state
            nextEncodedState = self.env.encodeCurrentState()
            # Append to the experience replay
            self.replay.add((currentEncodedState, action, reward, nextEncodedState, gameOver))
            # Go to the next player/agent...
            self.turnIndex = (self.turnIndex + 1) % len(self.agentPlayOrder)
            self.agentToPlay = self.agentPlayOrder[self.turnIndex]
            # If there is a game over, then reset the environment and players
            if gameOver:
                _ = self.env.reset()
                self.turnIndex = 0
                self.agentToPlay = self.agentPlayOrder[self.turnIndex]
            ##### TRAINING PHASE #####
            # Sample a batch from the experience replay, but only if the
            # buffer is full
            if len(self.replay.expSize) != self.replay.expSize:
                continue
            states, actions, rewards, nextStates, dones = self.replay.sample(batchSize)
            # Calculate the current Q-values to eventually
            # calculate the Q-values to train against...
            currentQVals = self.targetModel(states)
            nextQVals = self.predictionModel(nextStates)
            # Replace the correct Q-values in the matrix
            # with those of the target Q values from next states.
            # Not touching the other values in the matrix will
            # lead to a zero loss on there, preventing weights
            # from changing...
            nextQValMaxLocs = np.argmax(nextQVals, axis=1)
            expectedQVals = rewards + (1 - dones) * self.gamma * nextQValMaxLocs
            currentQVals[np.arange(currentQVals.shape[0]), actions] = expectedQVals
            # Train!!
            self.targetModel.train_on_batch()

