"""
File: environment.py
Creation Date: 2021-02-10
Location: base

This file contains the skeleton class of an environment. All future
environments are extended from this one. At the bare minimum, an
environment contains a reset method to return to the starting state.
"""

from .agents.basicAgent import Agent
from typing import Dict


class Environment:
    def __init__(self, agents: Dict[str, Agent], **kwargs):
        """
        This environment allows for multiple agents
        in the form of dictionaries. They should
        be mapped from agent name to the actual agent
        object
        :param agents: A dictionary mapping the agent
        name to the agent object.
        """
        self.agents = agents
        self.reset()
        return

    def reset(self):
        """
        Resets the environment (and the agent) and returns the
        starting state...
        :return: The starting state of the environment (not encoded)
        """
        raise NotImplementedError('Please implement a reset function for the environment!')

    def stepForward(self, agentName, action):
        """
        The main function to make a step in the environment.
        You must provide which agent's turn it is, and provide
        the action. It will perform the action, and return
        the resultant state, reward, and whether it resulted
        in a game over.
        :param agentName: The name of the agent, corresponding to the dictionary
        :param action: The action to take.
        :return: A 3-tuple of (resultant state, reward, game over flag)
        """
        raise NotImplementedError('Please implement a stepForward function for the environment!')

    def encodeCurrentState(self):
        raise NotImplementedError('Please implement an encoding of the current state!')

    def getNumberOfAgents(self):
        return len(self.agents.keys())