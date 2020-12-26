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
        self.maxScore = 0
        self.epsilon = 1
        self.learningRate = 0.01
        # Linear decay for epsilon, minimum of 0.1
        self.epsilonDecay = 0.03
        self.gamma = 0.99
        self.Qtable = np.random.random((2 ** 11, 3))
        self.env = SnakeGame(boardSize=boardSize)
        self.agent = SnakeAgent(environment=self.env)

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
        # Reset the agent
        self.agent.reset()
        while not self.agent.gameOver and len(self.agent.snakeFrames) < 10000:
            currentEncodedState = self.agent.encodeCurrentState()
            # With an epsilon% chance, choose
            # a random action. Otherwise, choose
            # the action with the largest Q-value.
            if random and np.random.rand() < self.epsilon:
                action = np.random.choice(self.agent.actionList)
            else:
                row = int(currentEncodedState, 2)
                rowData = self.Qtable[row]
                action = self.agent.actionList[np.argmax(rowData)]
            reward, gameOver = self.agent.makeMove(action)
        # Game is over, so return the game memory...
        if self.agent.score > self.maxScore:
            self.maxScore = self.agent.score
            self.env.exportGIF(f'BestGame.gif', self.agent.snakeFrames, scale=20)
        if makeGif:
            self.env.exportGIF(f'Game{self.gamesPlayed}.gif', self.agent.snakeFrames, scale=20)
            print(f'Game {self.gamesPlayed} scored {self.agent.score}! (Total states: {len(self.agent.snakeFrames)}, '
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
        for i in range(len(gameMemory) - 1):
            currState, turn, reward, gameOver = gameMemory[i]
            nextState = gameMemory[i + 1][0]
            currRow = int(currState, 2)
            currCol = self.agent.actionList.index(turn)
            nextRow = self.Qtable[int(nextState, 2)]
            maxNextQValue = max(nextRow)
            # Update, Q(s, a) = r(s, a) + gamma * maxNextQValue...
            self.Qtable[currRow, currCol] = reward + self.gamma * maxNextQValue
        # The last one is game over, there is no next state...
        goPrecState, turn, goReward, gameOver = gameMemory[-1]
        self.Qtable[int(goPrecState, 2), self.agent.actionList.index(turn)] = goReward  # Should be -1...
        # Decay the epsilon...
        self.epsilon *= (1 - self.epsilonDecay)

    def saveQTable(self, filename):
        np.savetxt(filename, self.Qtable, delimiter=', ')


if __name__ == '__main__':
    qtableObj = SnakeQTable()
    gamesToPlay = 1000
    for game in range(1, gamesToPlay + 1):
        print(f'\rGame {game}...', end='')
        gameMem = qtableObj.playGame()
        qtableObj.updateTable(gameMem)
    print('\nFinal game...')
    gameMem = qtableObj.playGame(makeGif=True, random=False)
    qtableObj.saveQTable(f'{gamesToPlay}Played.csv')


