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

from .agents.discreteAgent import DiscreteAgent
from .environment import Environment
from agents import *
from typing import List, Dict
import os


class QTable:
    def __init__(self, rows, cols, discreteAgents: Dict[str, List[DiscreteAgent]], environment: Environment,
                 epsilon=1, learningRate=0.1, epsilonDecay=0.005,
                 minEpsilon=0.01, gamma=0.95, **kwargs):
        self.gamesPlayed = 0
        self.maxScore = 0
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.epsilonDecay = epsilonDecay
        self.minEpsilon = minEpsilon
        self.gamma = gamma

        # Set up a QTable for each agent...
        self.QTables = {agentName: np.zeros((rows, cols), dtype=float) for agentName in environment.agents.keys()}
        self.env = Environment(discreteAgents, **kwargs)

    def mapStateToRow(self, encodedState):
        """
        This function allows for correct updating of the tables.
        Given a state, return a corresponding row in the table. It is
        up to the user to make sure there are no conflicts.
        :param encodedState:
        :return: The row of the table.
        """
        raise NotImplementedError('Please provide a mapping from the state to the row!')

    def playGame(self, firstTurn):
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
        # Set the agent to move first, a list
        # of all agents, and how many agents there
        # so we can loop through the list each time...
        # TODO: Provide agent turn order...
        allAgents = list(self.env.agents.keys())
        agentToPlay = firstTurn
        agentIndex = allAgents.index(agentToPlay)
        numOfAgents = self.env.getNumberOfAgents()
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
                action = self.env.agents[agentToPlay].chooseRandomAction()
            else:
                mappedRow = self.mapStateToRow(currentEncodedState)
                rowData = self.QTables[agentToPlay][mappedRow]
                action = self.


