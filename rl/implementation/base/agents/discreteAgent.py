"""
File: discreteAgent.py
Creation Date: 2020-02-13
Location: base/agents

This file contains an agent which would contain a
DISCRETE set of actions. When we implement a QTable, the agents
must be discrete, so we can select the best action. Continuous
agents can only be used with neural networks.
"""

from .basicAgent import Agent
import numpy as np


class DiscreteAgent(Agent):
    def __init__(self, actionList: list):
        """
        The constructor. Provide a list of actions to perform.
        Depending on your defined environment, they actions themselves
        can be of any type (strings, integers, etc.)
        :param actionList: The list of actions
        """
        super().__init__()
        self.actionList = actionList

    def reset(self):
        raise NotImplementedError('Please implement a reset() method!')

    def makeMove(self, action, env):
        raise NotImplementedError('Please implement a makeMove() method!')

    def checkValidAction(self, action):
        return action in self.actionList

    def chooseRandomAction(self):
        return np.random.choice(self.actionList)
