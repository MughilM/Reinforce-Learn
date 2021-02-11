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
        return

    def reset(self):
        raise NotImplementedError('Please implement a reset function for the agent!')
