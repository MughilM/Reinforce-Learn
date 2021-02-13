"""
File: agent.py
Creation Date: 2021-02-10
Location: BaseClasses

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

