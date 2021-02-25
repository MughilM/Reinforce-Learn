"""
File: game.py
Location: implementation/envs/snake
Creation Date: 2021-02-20

This file holds the rules for the game of the Snake. In order to pair it with
Q-table and neural network implementations of reinforcement learning, a custom
environment and agent have been added here.
The implementation shown here is sourced from the following site:
https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a
This is so I can test my RL system using a known state encoding.
Once teh tests have gone through, then I will transition to my own representation
of the snake state.
"""
from ...base.environment import Environment
from ...base.agents.basicAgent import Agent
from ...base.agents.discreteAgent import DiscreteAgent
from ...base.qtable import QTable

from typing import Dict
import copy


class SnakeEnv(Environment):
    def __init__(self, agents: Dict[str, Agent], boardSize=10, **kwargs):
        """
        Constructor. In addition to calling the standard Environment
        __init__, a user must also provide the size of the board. The board
        is a square, so provide the side length.
        :param agents: In snake, there is only one agent, but it must be
        given this way in order to comply with the Environment __init__
        :param boardSize: The side length of the board. This will be a square.
        :param kwargs: Additional keywords (Not necessary in Snake implementation)
        """
        super().__init__(agents, **kwargs)
        # A board size of less than 5 is too small to
        # learn anything meaningful.
        if boardSize < 5:
            raise ValueError(f'Board size of {boardSize} is too small!')
        self.boardSize = boardSize
        self.fruitLoc = (0, 0)
        self.placedFruit = False

    def reset(self):
        pass

    def stepForward(self, agentName, action):
        pass

    def encodeCurrentState(self):
        pass


class SnakeAgent(DiscreteAgent):
    def __init__(self, actionList: list):
        super().__init__(actionList)
        self.currentFrame = []
        self.score = 0
        # We need a mapping for the resultant direction given
        # the snake's current direction and the turn it
        # makes.
        self.TURN_RES = {
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

    def reset(self):
        """
        Clean the previous states, and put the snake in the
        top corner...This involves setting the starting
        frame, the starting direction (right), and the initial
        score of 3.
        :return:
        """
        self.currentFrame = [(0, 0), (0, 1), (0, 2)]
        self.score = 3
        self.direction = 'R'
        self.gameOver = False

    def makeTurn(self, turn, env):
        """
        If the snake needs to make a turn, this method
        is called. The environment is needed for the board
        size and the location of the fruit, so we can tell if we
        crashed (game over) or ate a fruit (snake gets longer).
        All the locations are updated automatically.
        :param turn: One of 'F', 'L', or 'R'
        :param env: A instance of SnakeEnv
        :return: The reward and whether it was a game over...
        """
        if turn not in self.actionList:
            raise ValueError(f'Action "{turn}" not in the action list!')
        if self.gameOver:
            print('Game is over! Please reset!')
            return
        newState = copy.deepcopy(self.currentFrame)
        newDirection = self.TURN_RES[self.direction][turn]
        for i in range(self.score - 1):
            newState[i] = newState[i + 1]
        # If we didn't change direction, push everything one....
        if newDirection == self.direction:
            # Using the current direction, update the head...
            if self.direction == 'U':
                newState[-1] = (newState[-1][0] - 1, newState[-1][1])
            elif self.direction == 'D':
                newState[-1] = (newState[-1][0] + 1, newState[-1][1])
            elif self.direction == 'L':
                newState[-1] = (newState[-1][0], newState[-1][1] - 1)
            else:
                newState[-1] = (newState[-1][0], newState[-1][1] + 1)
        # Changed direction.
        else:
            if newDirection == 'U':
                newState[-1] = (newState[-2][0] - 1, newState[-2][1])
            elif newDirection == 'D':
                newState[-1] = (newState[-2][0] + 1, newState[-2][1])
            elif newDirection == 'L':
                newState[-1] = (newState[-2][0], newState[-2][1] - 1)
            else:
                newState[-1] = (newState[-2][0], newState[-2][1] + 1)
        # Check to see if we've crashed...
        # Either we ate ourself or went out of bounds.
        # Give ourselves a -10 reward...
        if (newState[-1] in newState[:-1]) or \
                (any(r < 0 or c < 0 or r >= env.boardSize or c >= env.boardSize for r, c in newState)):
            self.gameOver = True
            reward = -10
        # Check to see if we've eaten a fruit.
        # Use the tail location of the previous
        # state to extend. Takes care of weird edge cases.
        elif newState[-1] == env.fruitLoc:
            newState.insert(0, self.currentFrame[0])
            self.score += 1
            reward = 10
        else:
            reward = 0  # We didn't crash or eat, so no reward
        self.direction = newDirection  # Set to new direction...
        self.currentFrame = newState
        return reward, self.gameOver


class SnakeQTable(QTable):
    def __init__(self, outputDir, experimentName, rows, cols, discreteAgents: Dict[str, DiscreteAgent],
                 environment: Environment, stateLimit=10000,
                 epsilon=1, learningRate=0.1, epsilonDecay=0.995,
                 minEpsilon=0.01, gamma=0.95, overwrite=False):
        super().__init__(outputDir, experimentName, rows, cols, discreteAgents, environment,
                         stateLimit=stateLimit, epsilon=epsilon, learningRate=learningRate, epsilonDecay=epsilonDecay,
                         minEpsilon=minEpsilon, gamma=gamma, overwrite=overwrite)

    def mapStateToRow(self, encodedState):
        """
        For this implementation of the snake's QTable, the state is encoded as an
        11-digit binary string, so just convert this to an integer for our row index.
        :param encodedState: The encoded state returned from SnakeEnv. Should be an 11-digit bit string.
        :return: The row index.
        """
        return int(encodedState, 2)

    def loadDataArtifacts(self, **kwargs):
        """
        In `saveDataArtifacts` we saved a list of the scores and a
        plot. We will just load the score list because we
        can't do much with a raw plot image.
        :param kwargs:
        :return:
        """
        pass

    def saveDataArtifacts(self, **kwargs):
        """
        For extra artifacts, we can save a list of the scores,
        along with a plot across the number of the games.
        :param kwargs:
        :return:
        """
        pass