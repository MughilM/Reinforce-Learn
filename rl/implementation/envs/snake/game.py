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
import altair as alt
from pandas import DataFrame
import numpy as np
from itertools import product

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

    def makeMove(self, action, env):
        """
        If the snake needs to make a action, this method
        is called. The environment is needed for the board
        size and the location of the fruit, so we can tell if we
        crashed (game over) or ate a fruit (snake gets longer).
        All the locations are updated automatically.
        :param action: One of 'F', 'L', or 'R'
        :param env: A instance of SnakeEnv
        :return: The reward and whether it was a game over...
        """
        if action not in self.actionList:
            raise ValueError(f'Action "{action}" not in the action list!')
        if self.gameOver:
            print('Game is over! Please reset!')
            return
        newState = copy.deepcopy(self.currentFrame)
        newDirection = self.TURN_RES[self.direction][action]
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


class SnakeEnv(Environment):
    def __init__(self, agents: Dict[str, SnakeAgent], boardSize=10, **kwargs):
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
        self.agents = agents
        # A board size of less than 5 is too small to
        # learn anything meaningful.
        if boardSize < 5:
            raise ValueError(f'Board size of {boardSize} is too small!')
        self.boardSize = boardSize
        self.fruitLoc = (0, 0)
        self.placedFruit = False
        # Save the agent name, since there's only one, and
        # we need to easily access it.
        self.agentName = agents.keys()[0]
        self.reset()

    def placeFruit(self, snakeLocs):
        """
        Depending on where the snake, this method
        randomly places the fruit somewhere on the
        board.
        :param snakeLocs: The current frame of the snake,
        should be the frame from the agent...
        :return:
        """
        validLocs = [(r, c) for r, c in product(range(self.boardSize), repeat=2)
                     if (r, c) not in snakeLocs]
        # Randomly select one...
        selectionIndex = np.random.choice(len(validLocs))
        self.fruitLoc = validLocs[selectionIndex]
        self.placedFruit = True
        return

    def reset(self):
        """
        Resets the board....and a new fruit
        is placed somewhere at random. The initial
        location of the snake is top right corner facing
        the left....
        :return: The starting state of the board.
        """
        self.agents[self.agentName].reset()
        # Place the fruit according to the current frame...
        self.placeFruit(self.agents[self.agentName].currentFrame)
        startState = {
            'boardSize': self.boardSize,
            'snakeLocs': self.agents[self.agentName].currentFrame,
            'fruitLoc': self.fruitLoc
        }
        return startState

    def stepForward(self, agentName, action):
        """
        For the snake, this will go one step forward, according to the action of
        forward ('F'), left ('L'), or right ('R'). As usual, the next state,
        obtained reward, and whether it's a game over will be returned.
        :param agentName: The name of agent. There's only one here...
        :param action: One of ['F', 'L', 'R']
        :return: A 3-tuple of the next state, reward, and game over
        """
        if self.agents[agentName].gameOver:
            raise ValueError('Game is already over. Please reset!')
        reward, gameOver = self.agents[agentName].makeMove(action, env=self)
        # Check to see if the snake ate the fruit and grew...
        if reward > 0:
            self.placeFruit(self.agents[agentName].currentFrame)
        # The new state gets returned as a dictionary (it's unencoded)
        newState = {
            'boardSize': self.boardSize,
            'snakeLocs': self.agents[agentName].currentFrame,
            'fruitLoc': self.fruitLoc
        }
        return newState, reward, gameOver

    def encodeCurrentState(self):
        """
        This method turns the dictionary-like state into a state
        encoded with 11 bits.
        - Is there immediate danger in front, left, or right of the snake? (3 bits)
        - The direction of the fruit wrt to the snake (up, down, left, right) (4 bits)
        - The direction of the snake (up, down, left, right) (4 bits)
        :return: The 11-bit string that is an encoding of the current environment state.
        """
        directionCode = {
            'U': '1000',
            'D': '0100',
            'L': '0010',
            'R': '0001'
        }
        coding = ''
        # Look at the snake end for immediate danger. Depending
        # on the snake direction, the 3 cells we need to check will differ.
        head = self.agents[self.agentName].currentFrame[-1]
        snakeDirection = self.agents[self.agentName].direction
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
        # For each of the three proximity cells, check if they are
        # part of the snake body or off the board...
        dangers = ((r, c) in self.agents[self.agentName].currentFrame or
                   not (0 <= r < self.boardSize and 0 <= c < self.boardSize)
                   for r, c in proximity)
        # Convert the booleans to 0s and 1s
        coding += ''.join(map(lambda x: str(int(x)), dangers))
        # Add the fruit bits...
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
        # Now add the direction according to the defined dictionary
        coding += directionCode[snakeDirection]
        return coding


class SnakeQTable(QTable):
    def __init__(self, outputDir, experimentName, rows, cols, discreteAgents: Dict[str, SnakeAgent],
                 environment: Environment, stateLimit=10000,
                 epsilon=1, learningRate=0.1, epsilonDecay=0.995,
                 minEpsilon=0.01, gamma=0.95, overwrite=False):
        super().__init__(outputDir, experimentName, rows, cols, discreteAgents, environment,
                         stateLimit=stateLimit, epsilon=epsilon, learningRate=learningRate, epsilonDecay=epsilonDecay,
                         minEpsilon=minEpsilon, gamma=gamma, overwrite=overwrite)
        # An array to store the snake scores. During saving, this list will
        # be saved and then loaded if an experiment needs to be continued.
        # A plot will also be saved.
        self.snakeScores = []

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
        # Load the snake scores from the text file...
        self.snakeScores = np.loadtxt('scores.txt')

    def saveDataArtifacts(self, **kwargs):
        """
        For extra artifacts, we can save a list of the scores,
        along with a plot across the number of the games.
        :param kwargs:
        :return:
        """
        # First plot the scores across the games
        # and save it.
        data = DataFrame(
            {'Game': np.arange(1, len(self.snakeScores) + 1)},
            {'Score': self.snakeScores}
        )
        chart = alt.Chart(data).mark_line().encode(
            x='Game:Q',
            y='Score:Q'
        )
        chart.save('scores.png')
        # Then save the score values themselves...
        np.savetxt('scores.txt', self.snakeScores)

    def updateGameMetrics(self, gameMemory):
        """
        In the case of the snake game, all we are doing is
        adding the score to our list.
        :param gameMemory:
        :return:
        """
        # We start with a score of 3. Every time the snake eats a fruit, it is
        # rewarded. Thus, we count all the times it got rewarded and then add 3
        # to get the final score...
        score = 3
        for _, _, _, reward, _, _ in gameMemory:
            if reward > 0:
                score += 1
        self.snakeScores.append(score)
