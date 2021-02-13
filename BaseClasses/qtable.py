"""
File: qtable.py
Creation Date: 2021-02-13
Location: BaseClasses

This file contains the base implementation of a QTable.
By default, each agent has its own table. Two things
that must be provided are the shapse of the tables and
a way to map the encoded state returned by the environment
to the row of the table. The goal of this class is for
future games that use the QTables to provide as few
information as possible.
"""

import numpy as np
from .environment import Environment
from .agent import Agent
from typing import List
import os


class QTable:
    def __init__(self, rows, cols, environment: Environment, epsilon=1, learningRate=0.1, epsilonDecay=0.005,
                 minEpsilon=0.01, gamma=0.95):
        self.gamesPlayed = 0
        self.maxScore = 0
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.epsilonDecay = epsilonDecay
        self.minEpsilon = minEpsilon
        self.gamma = gamma

        # Set up a QTable for each agent...
        self.QTables = {agentName: np.zeros((rows, cols), dtype=float) for agentName in environment.agents.keys()}
        self.env = environment

    def playGame(self):
        """
        Plays a game according to the steps defined in the
        environment's stepForward. This method is designed to be
        environment-agnostic, meaning changing the rules will change
        the behaviour. This will also return the encoded game
        memory.
        :return: A list of [state, agentName, action, reward, nextState, gameOver]
        for however long the game lasts...
        """
        self.gamesPlayed += 1
        # Keep a counter for number of states, to prevent
        # continuous looping in a game...
        stateCounts = 1
        # Reset the environment...
        currentState = self.env.reset()

        gameOver = False
        gameMemory: List[List] = []
        while not gameOver and stateCounts < 10000:
            currentEncodedState = self.env.encodeCurrentState()
            # If it's at least the second state, then it was the "next state"
            # of the previous state/action...
            if len(gameMemory) > 0:
                gameMemory[-1][4] = currentEncodedState
            # With an epsilon % chance, choose a random action.
            # Otherwise, choose the action with the largest Q-value..
            if np.random.rand() < self.epsilon:
                action =


