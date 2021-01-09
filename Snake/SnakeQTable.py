"""
File: SnakeQTable.py
Location: /Snake/
Creation Date: 2020-09-30

This file implements the Q-table for the
snake game. The states are coded as 11-bit
strings, and there are only 3 actions
(forward, left, and right). Thus, the Q-table
is a pretty straightforward 2^11 by 3 table.
"""

import numpy as np
from .SnakeAgent import SnakeAgent
from .SnakeEnv import SnakeGame
from .utils import *

import numpy as np

class SnakeQTable:
    def __init__(self, boardSize=20):
        self.gamesPlayed = 0
        self.maxScore = 0
        self.epsilon = 1
        self.learningRate = 0.05
        # Linear decay for epsilon, minimum of 0.1
        self.epsilonDecay = 0.0005
        self.minEpsilon = 0.01
        self.gamma = 0.9
        self.Qtable = np.random.random((2 ** 11, 3))
        self.agent = SnakeAgent()
        self.env = SnakeGame(snakeAgent=self.agent, boardSize=boardSize)

    def playGame(self, makeGif=False, random=True):
        """
        Plays a game of snake until a game over is reached.
        Exploration and exploitation based on the
        epsilon is implemented to select actions.
        The epsilon also decays from one game to the
        next.
        :return: The encoded game memory.
        """
        self.gamesPlayed += 1
        stateCounts = 1
        # Reset the agent
        currentState = self.env.reset()
        # Only save the frames if we're making a GIF...
        allSnakeStates = []
        if makeGif:
            allSnakeStates = [produceBoardFrame(snakeGameState=currentState)]  # One frame with the first state...
        gameOver = False
        while not gameOver and stateCounts < 10000:
            currentEncodedState = self.env.encodeCurrentState()
            # With an epsilon% chance, choose
            # a random action. Otherwise, choose
            # the action with the largest Q-value.
            if random and np.random.rand() < self.epsilon:
                action = np.random.choice(self.agent.actionList)
            else:
                row = int(currentEncodedState, 2)
                rowData = self.Qtable[row]
                action = self.agent.actionList[np.argmax(rowData)]
            currentState, reward, gameOver = self.env.stepForward(action)
            if makeGif:
                allSnakeStates.append(produceBoardFrame(currentState))
            stateCounts += 1
        # Game is over, so return the game memory...
        if self.agent.score > self.maxScore:
            self.maxScore = self.agent.score
        if makeGif:
            exportGIF(frames=allSnakeStates, filename=f'Game{self.gamesPlayed}.gif', scale=15)
            print(f'Game {self.gamesPlayed} scored {self.agent.score}! '
                  f'Best Score: {self.maxScore})')
        return self.agent.getGameMemory()

    def updateTable(self, gameMemory):
        """
        Given a game memory, this will update the Q-table based
        on Bellman's equation.
        :param gameMemory: A single game's memory. Assumed to end
        in a game over.
        :return:
        """
        for currState, turn, reward, nextState, gameOver in gameMemory:
            currRow = int(currState, 2)
            currCol = self.agent.actionList.index(turn)
            nextRow = self.Qtable[int(nextState, 2)]
            maxNextQValue = max(nextRow)
            # Update, Q(s, a) = Q(s, a) + alpha * ( r(s, a) + gamma * maxNextQValue - Q(s,a) )...
            self.Qtable[currRow, currCol] += self.learningRate * (reward + self.gamma * maxNextQValue -
                                                                  self.Qtable[currRow, currCol])
        # Decay the epsilon...
        self.epsilon = max(self.epsilon * (1 - self.epsilonDecay), self.minEpsilon)

    def saveQTable(self, filename):
        np.savetxt(filename, self.Qtable, delimiter=', ')


if __name__ == '__main__':
    qtableObj = SnakeQTable()
    gamesToPlay = 5
    for game in range(1, gamesToPlay + 1):
        print(f'\rGame {game}...', end='')
        gameMem = qtableObj.playGame()
        qtableObj.updateTable(gameMem)
    print('\nFinal game...', 'Current epsilon is', qtableObj.epsilon)
    gameMem = qtableObj.playGame(makeGif=True, random=False)
    qtableObj.saveQTable(f'{gamesToPlay}Played.csv')


