##################################################
# File: SnakeQTable.py
# Author: Mughil Pari
# Location: /Snake/
#
# This file implements the Q-table for the
# snake game. The states are coded as 11-bit
# strings, and there are only 3 actions
# (forward, left, and right). Thus, the Q-table
# is a pretty straightforward 2^11 by 3 table.
##################################################

import numpy as np
from .SnakeAgent import SnakeAgent
from .SnakeEnv import SnakeGame

class SnakeQTable:
    def __init__(self, boardSize=10):
        self.gamesPlayed = 0
        self.epsilon = 0.95
        # Linear decay for epsilon, minimum of 0.1
        self.epsilonDecay = 0.05
        self.gamma = 0.99
        self.Qtable = np.zeros((2 ** 11, 3), dtype=float)
        self.env = SnakeGame(boardSize=boardSize)
        self.agent = SnakeAgent(environment=self.env)

    def playGame(self):
        """
        Plays a game of snake until a game over is reached.
        Exploration and exploitation based on the
        epsilon is implemented to select actions.
        The epsilon also decays from one game to the
        next.
        :return:
        """
        # Reset the agent
        self.agent.reset()
        while not self.agent.gameOver:
            # With an epsilon% chance, choose
            # a random action. Otherwise, choose
            # the action with the largest Q-value.
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.agent.actionList)

