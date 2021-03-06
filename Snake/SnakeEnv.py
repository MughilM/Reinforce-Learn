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
        self.fruitLoc = (0, 0)
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

    def encodeCurrentState(self):
        """
        Primarily internal method. It will take the current
        state of the snake, and encode it according to our rules.
        It will use the most recent location of the snake in
        the environment variables.
        :return: The state coded as an 11-bit string:
            - Is there immediate danger in front, left,
            or right of the snake?
            - The direction of the fruit (up, down, left,
            right). From a top-down perspective. More
            than one is possible.
            - The direction of the snake (up, down, left,
            right)
        Thus, the coding is [danger ==> 'FLR']
        [fruit direction ==> 'UDLR']
        [snake direction ==> 'UDLR'] (mutually exclusive)
        """
        directionCode = {
            'U': '1000',
            'D': '0100',
            'L': '0010',
            'R': '0001'
        }
        coding = ''
        # For immediate danger, we look at the snake head, and see
        # if either the edge of the board or a snake body part is
        # next to it. The array is in FLR order.
        head = self.agent.currentFrame[-1]
        snakeDirection = self.agent.direction
        if snakeDirection == 'U':
            proximity = [
                (head[0] - 1, head[1]),
                (head[0], head[1] - 1),
                (head[0], head[1] + 1)
            ]
        elif snakeDirection == 'D':
            proximity = [
                (head[0] + 1, head[1]),
                (head[0], head[1] + 1),
                (head[0], head[1] - 1)
            ]
        elif snakeDirection == 'L':
            proximity = [
                (head[0], head[1] - 1),
                (head[0] + 1, head[1]),
                (head[0] - 1, head[1])
            ]
        else:
            proximity = [
                (head[0], head[1] + 1),
                (head[0] - 1, head[1]),
                (head[0] + 1, head[1])
            ]
        # Lotta stuff going on here:
        #   Check if each location is in the snake body
        #   or off the board. Convert the Trues and Falses
        # into a bit string we can directly attach to our coding.
        dangers = ((r, c) in self.agent.currentFrame or not (0 <= r < self.boardSize and 0 <= c < self.boardSize)
                   for r, c in proximity)
        coding += ''.join(map(lambda x: str(int(x)), dangers))
        # Now the fruit location. The fruit can't be both above and
        # below the snake, so append in pairs.
        fruitR, fruitC = self.fruitLoc
        if head[0] > fruitR:
            coding += '10'
        elif head[0] < fruitR:
            coding += '01'
        else:
            coding += '00'  # The fruit is on the same row
        # Left/right
        if head[1] > fruitC:
            coding += '10'
        elif head[1] < fruitC:
            coding += '01'
        else:
            coding += '00'

        # Now the direction of the snake...Straightforward...
        coding += directionCode[snakeDirection]
        return coding