"""
File: basicAgent.py
Creation Date: 2021-02-10
Location: base

This file contains the bare implementation of an agent. All future agents
should extend from this class. At minimum, an agent needs a reset function,
and maybe a couple of properties, although that is up to the user.
"""


class Agent:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        raise NotImplementedError('Please implement a reset function for the agent!')

    def makeMove(self, action, env):
        return

    def checkValidAction(self, action):
        """
        This function takes an action and verifies to make sure
        the action is allowed. This could be as simple as checking
        a defined actionList (if we have a discrete action space).
        :param action: The action to take.
        :return: True/False, if the action is allowed/not allowed.
        """
        raise NotImplementedError('Please provide a way to verify a provided action!')

    def chooseRandomAction(self):
        """
        This method provides custom functionality on what "choosing a random
        action" entails. If it's a discrete action space, it would simply be
        `np.random.choice`. But if it's continuous, then it may entail choosing
        a random number between 0 and 1 for example.
        :return: The randomly chosen action.
        """
        raise NotImplementedError('Please provide a way to randomly choose an action!')
