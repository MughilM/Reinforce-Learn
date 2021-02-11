"""
File: environment.py
Creation Date: 2021-02-10
Location: BaseClasses

This file contains the skeleton class of an environment. All future
environments are extended from this one. At the bare minimum, an
environment contains a reset method to return to the starting state.
"""


class Environment:
    def __init__(self):
        return

    def reset(self):
        raise NotImplementedError('Please implement a reset function for the environment!')
