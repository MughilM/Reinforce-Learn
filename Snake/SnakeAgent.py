##################################################
# File: SnakeAgent.py
# Author: Mughil Pari
# Location: /Snake/
#
# This file implements the snake agent
# (Hello, this is Snake) for the snake game. This
# has the methods which actually play the game.
# The actual snake mechanices are implemented
# here, while the environment simply houses the
# board and the food.
##################################################

import numpy as np
from .playSnake import SnakeGame

import copy

class SnakeAgent:
    def __init__(self, environment: SnakeGame):
        self.env = environment
        self.prevStates = []
        self.currState = []
        self.score = 0
        # Direction is from the snake's
        # perspective. Direction itself
        # is up, down, left, right. But
        # snake's turning is from the
        # snake's perspective e.g.
        # moving down and turning left
        # means the snake is now going RIGHT.
        self.DIR_RESULT = {
            'U': {
                'F': 'U',
                'L': 'L',
                'R': 'R'
            },
            'D': {
                'F': 'D',
                'L': 'R',
                'R': 'L'
            },
            'L': {
                'F': 'L',
                'L': 'D',
                'R': 'U'
            },
            'R': {
                'F': 'R',
                'L': 'U',
                'R': 'D'
            }
        }
        self.direction = ''
        self.gameOver = 0
        self.reset()

    def reset(self):
        # Reset the environment
        self.env.reset()
        # Clean the previous states,
        # and put the snake in the top corner...
        self.prevStates = []
        self.currState = [(0, 0), (0, 1), (0, 2)]
        self.score = 3
        self.direction = 'R'
        self.gameOver = 0
        self.env.placeFruit(self.currState)

    def makeMove(self, turn):
        """
        The meat method. Given a turn ('F', 'L', 'R'),
        moves the snake in that direction by one
        step and returns appropriate reward. A reward of 1 is
        given if it eats a fruit, -1 if the snake dies, and
        0 otherwise. The locations are all updated automatically.
        If the move leads to a fruit, it eats the fruit,
        the snake gets longer,
        :param turn:
        :return:
        """
        if self.gameOver:
            print('Game is over! Please reset!')
            return
        self.prevStates.append(copy.deepcopy(self.currState))
        newDirection = self.DIR_RESULT[self.direction][turn]
        for i in range(self.score - 1):
            self.currState[i] = self.currState[i + 1]
        # If we didn't change direction, push everything one...
        if newDirection == self.direction:
            # Using the current direction, update the head...
            if self.direction == 'U':
                self.currState[-1] = (self.currState[-1][0] - 1, self.currState[-1][1])
            elif self.direction == 'D':
                self.currState[-1] = (self.currState[-1][0] + 1, self.currState[-1][1])
            elif self.direction == 'L':
                self.currState[-1] = (self.currState[-1][0], self.currState[-1][1] - 1)
            else:
                self.currState[-1] = (self.currState[-1][0], self.currState[-1][1] + 1)
        # Changed direction.
        else:
            if newDirection == 'U':
                self.currState[-1] = (self.currState[-2][0] - 1, self.currState[-2][1])
            elif newDirection == 'D':
                self.currState[-1] = (self.currState[-2][0] + 1, self.currState[-2][1])
            elif newDirection == 'L':
                self.currState[-1] = (self.currState[-2][0], self.currState[-2][1] - 1)
            else:
                self.currState[-1] = (self.currState[-2][0], self.currState[-2][1] + 1)

        # Check to see if we've crashed...
        # Either we ate ourself or went out of bounds.
        if (self.currState[-1] in self.currState[:-1]) or \
                (any(r < 0 or c < 0 or r >= self.env.boardSize or c >= self.env.boardSize for r, c in self.currState)):
            self.gameOver = True
            return -1, self.gameOver
        # Check to see if we've eaten a fruit.
        # Use the tail location of the previous
        # state to extend. Takes care of weird edge cases.
        if self.currState[-1] == self.env.fruitLoc:
            self.currState.insert(0, self.prevStates[-1][0])
            return 1, self.gameOver
        # We didn't crash or eat, so no reward
        return 0, self.gameOver

