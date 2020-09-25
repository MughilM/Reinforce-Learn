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
from .SnakeEnv import SnakeGame

import copy


class SnakeAgent:
    def __init__(self, environment: SnakeGame):
        self.env = environment
        self.states = []
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
        self.gameOver = False
        self.reset()

    def reset(self):
        # Reset the environment
        self.env.reset()
        # Clean the previous states,
        # and put the snake in the top corner...
        self.states = [[(0, 0), (0, 1), (0, 2)]]
        self.score = 3
        self.direction = 'R'
        self.gameOver = False
        self.env.placeFruit(self.states[-1])

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
        currState = copy.deepcopy(self.states[-1])
        newDirection = self.DIR_RESULT[self.direction][turn]
        for i in range(self.score - 1):
            currState[i] = currState[i + 1]
        # If we didn't change direction, push everything one...
        if newDirection == self.direction:
            # Using the current direction, update the head...
            if self.direction == 'U':
                currState[-1] = (currState[-1][0] - 1, currState[-1][1])
            elif self.direction == 'D':
                currState[-1] = (currState[-1][0] + 1, currState[-1][1])
            elif self.direction == 'L':
                currState[-1] = (currState[-1][0], currState[-1][1] - 1)
            else:
                currState[-1] = (currState[-1][0], currState[-1][1] + 1)
        # Changed direction.
        else:
            if newDirection == 'U':
                currState[-1] = (currState[-2][0] - 1, currState[-2][1])
            elif newDirection == 'D':
                currState[-1] = (currState[-2][0] + 1, currState[-2][1])
            elif newDirection == 'L':
                currState[-1] = (currState[-2][0], currState[-2][1] - 1)
            else:
                currState[-1] = (currState[-2][0], currState[-2][1] + 1)
        self.direction = newDirection
        # Check to see if we've crashed...
        # Either we ate ourself or went out of bounds.
        if (currState[-1] in currState[:-1]) or \
                (any(r < 0 or c < 0 or r >= self.env.boardSize or c >= self.env.boardSize for r, c in currState)):
            self.gameOver = True
            self.states.append(currState)
            return -1, self.gameOver
        # Check to see if we've eaten a fruit.
        # Use the tail location of the previous
        # state to extend. Takes care of weird edge cases.
        if currState[-1] == self.env.fruitLocs[-1]:
            currState.insert(0, self.states[-1][0])
            self.score += 1
            self.env.placeFruit(currState)
            self.states.append(currState)
            return 1, self.gameOver
        self.states.append(currState)
        # We didn't crash or eat, so no reward
        return 0, self.gameOver

