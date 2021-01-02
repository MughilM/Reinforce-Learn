"""
File: SnakeEnv.py
Location: /Snake/
Creation Date: 2020-09-17

This file implements the simple environment
for the snake game. It only has the game board's
properties, because the snake can be considered an
agent himself. Thus, for the environment, we only
store the size of the board and the location of the
fruit. It also provides capability to export the
played game as a GIF, given the locations
of the snake at each frame.
"""

import numpy as np
from itertools import product
import imageio
import os
from .SnakeAgent import SnakeAgent


class SnakeGame:
    def __init__(self, snakeAgent: SnakeAgent, boardSize=10):
        """
        Saves the board size and randomly
        places the fruit. Same functionality
        as reset(), except the board size
        is saved. Board size larger than 5 is
        required.
        :param boardSize: Side length of board
        """
        self.agent = snakeAgent
        # Force a minimum size of 5...
        if boardSize < 5:
            raise ValueError("Board size of {} is too small!".format(boardSize))
        self.boardSize = boardSize
        # Don't pay attention to the values here.
        # They'll get reset. It's just so my IDE can
        # recognize the data types used :)
        self.fruitLoc = ()
        self.placedFruit = False
        self.reset()

    def reset(self):
        """
        Resets the board along with the agent.
        A new fruit is placed on the empty board...
        :return: The starting state of the environment
        """
        self.agent.reset()
        self.placeFruit(self.agent.currentFrame)
        startState = {
            'boardSize': self.boardSize,
            'snakeLocs': self.agent.currentFrame,
            'fruitLoc': self.fruitLoc
        }
        return startState

    def placeFruit(self, snakeLocs):
        """
        Places the fruit at random depending on the
        location of the snake. You should call this
        from the agent right after resetting and right
        after a fruit is eaten.
        :param snakeLocs:
        :return:
        """
        validLocs = [(r, c) for r, c in product(range(self.boardSize), repeat=2)
                     if (r, c) not in snakeLocs]
        # Randomly select one...
        selectionIndex = np.random.choice(len(validLocs))
        self.fruitLoc = validLocs[selectionIndex]
        self.placedFruit = True
        return

    def stepForward(self, action):
        """
        This function steps forward one time step in the environment.
        It will use the given action and apply it to the contained agent above.
        The agent will return the reward and whether it resulted in a game over.
        The environment's state is also passed into the function to use extra
        variables. New fruit placement is done here, not in the makeMove()
        function.
        :param action: The action to take...One of 'F', 'L', 'R'
        :return: The new state (as dictionary), reward, and game over.
        The new state is like {'snakeLocs': ..., 'fruit loc': ...}. Any
        preprocessing that is needed for, say, Q-learning should be done
        separately...
        """
        if self.agent.gameOver:
            raise ValueError('Game is already over. Please reset!')
        reward, gameOver = self.agent.makeMove(action, env=self)
        # We check to see if the snake grow by looking at the reward...
        if reward > 0:
            self.placeFruit(self.agent.currentFrame)
        # Return the new state as dictionary, along with reward and game over...
        newState = {
            'boardSize': self.boardSize,
            'snakeLocs': self.agent.currentFrame,
            'fruitLoc': self.fruitLoc
        }
        return newState, reward, gameOver

    def exportGIF(self, filename: str, frames: list, scale=1):
        """
        Takes the previous states and exports them
        as a playable GIF. Automatically places
        it in the ./Data/gifs/ folder.
        :param: filename: Name of GIF file. Automatically
        prepended with ./Data/ and adds .gif if not there
        :param: frames: A list containing the locations
        of the snake at each frame. Expects each element
        to be a list of two-tuples. The last element
        is the snake head.
        :param: scale: How much to scale up the GIF. By
        default, each "cell" takes up 1 pixel.
        :return:
        """
        if not filename.endswith('.gif'):
            filename += '.gif'
        playedGame = np.zeros(shape=(len(frames), (self.boardSize + 2) * scale, (self.boardSize + 2) * scale),
                              dtype=np.uint8)
        filename = os.path.join('./Snake/Data/gifs', filename)
        # The fruit locations are only added every time the
        # snake eats, so we have to see when it eats, given
        # when the snake gets longer...
        with imageio.get_writer(filename, mode='I') as writer:
            for i, frame in enumerate(frames):
                # Convert the frame to an image-like ndarray...
                playedGame[i] = self.produceBoardFrame(frame, scale=scale)
                # Append to the writer...
                writer.append_data(playedGame[i])

    def boardToString(self, snakeLocs: list):
        """
        Given the locations of the snake, prints
        out the board.
        :param snakeLocs: The locations of the snake.
        :return:
        """
        gameStr = np.zeros(shape=(self.boardSize + 2, self.boardSize + 2), dtype=str)
        gameStr[:, :] = '-'
        # Hashtags for board border...
        gameStr[0, :] = '#'
        gameStr[-1, :] = '#'
        gameStr[:, 0] = '#'
        gameStr[:, -1] = '#'
        # 's' for snake body, 'h' for haed, 'f' for fruit
        gameStr[self.fruitLocs[-1][0] + 1, self.fruitLocs[-1][1] + 1] = 'f'
        for r, c in snakeLocs[:-1]:
            gameStr[r + 1, c + 1] = 's'
        gameStr[snakeLocs[-1][0] + 1, snakeLocs[-1][1] + 1] = 'h'
        return '\n'.join(''.join(row) for row in gameStr)
