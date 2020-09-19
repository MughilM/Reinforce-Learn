##################################################
# File: playSnake.py
# Author: Mughil Pari
# Location: /Snake/
#
# This file implements the rules of the game Snake.
# It includes methods to make moves, and also to
# compile the entire game into an animated gif
# for future viewing. This provides an environment
# for which an agent can act upon. The state
# of the game will be stored in matrices.
# The environment simply stores the states
# of the game and provides capability of
# exporting the states into a playable GIF.
##################################################

import numpy as np
import imageio

import os


class SnakeGame:
    def __init__(self, boardSize=10):
        """
        Creates the board, along with score, and random placement
        of fruit. Also creates the export
        directories if not present.
        :param boardSize: Side length of board
        """
        # Create directories...
        os.makedirs('./Data/imgs/', exist_ok=True)
        os.makedirs('./Data/gifs/', exist_ok=True)
        # Force a minimum size of 5...
        if boardSize < 5:
            raise ValueError("Board size of {} is too small!".format(boardSize))
        self.boardSize = boardSize
        # Don't pay attention to the values here.
        # They'll get reset. It's just so my IDE can
        # recognize the data types used :)
        self.score = 0
        self.prevStates = []
        self.currState = np.array([], dtype=float)
        self.reset()

    def reset(self):
        """
        Resets the board. Randomly places first fruit and
        places snake in top corner
        :return:
        """
        self.score = 3
        self.prevStates = []
        # We can make the initial position of the
        # snake be in the top corner...
        self.currState = np.zeros((self.boardSize, self.boardSize), dtype=float)
        self.currState[0, :self.score] = 1
        # The head of the snake will be slightly darker...
        self.currState[0, self.score - 1] = 0.7
        # Randomly place the fruit in the space...
        fruitLoc = np.random.randint(low=self.score, high=self.boardSize ** 2)
        self.currState[fruitLoc // self.boardSize, fruitLoc % self.boardSize] = 0.4

    def exportGIF(self, filename: str):
        """
        Takes the previous states and exports them
        as a playable GIF. Automatically places
        it in the ./Data/gifs/ folder.
        :param: filename: Name of GIF file. Automatically
        prepended with ./Data/ and adds .gif if not there
        :return:
        """
        if not filename.endswith('.gif'):
            filename += '.gif'
        filename = os.path.join('./Data/', filename)
        with imageio.get_writer(filename, mode='I') as writer:
            for state in self.prevStates:
                writer.append_data(state)
