"""
File: snakeMain.py
Creation Date: 2021-03-27
Location; rl/implementation

This file houses the main function for just the snake game. It is intended to ensure all the
code works through without issues. Additionally, I planned to have one file that
would run any game implementation, but I was unsure of the details of what parameters
go where, as well as how to deal with parameters that are specific to each game.
The code in the this file would provide a general fromework on how to pass
the arguments through. It will include both a Qtable and DQN approach.
"""

from .config import *
from .envs.snake.game import *
import os

# First let's do the snake Q table approach...
# Define some variables...
boardSize = 10  # Start small...
experimentName = 'standalone_exp'
initialE = 1
lr = 0.15
stateLimit = 5000
epsilonDecay = 0.99
minEpsilon = 0.01
gamma = 0.99
overwrite = True  # We need to monitor changes in the input...
episodes = 2000
# To run a Qtable approach, we have to create the
# agent and environment, and provide them as parameters
# to the Qtable class...
snakeAgent = SnakeAgent(actionList=['F', 'L', 'R'])
snakeEnv = SnakeEnv({'snake': snakeAgent}, boardSize=boardSize)

# Now create the qtable. The output directory is EXP_OUTPUT_DIR
# from config, the rows are 2^11, the number of columns are 3
snakeTable = SnakeQTable(
    outputDir=EXP_OUTPUT_DIR,
    experimentName=experimentName,
    rows=2 ** 11,
    cols=3,
    discreteAgents={'snake': snakeAgent},
    environment=snakeEnv,
    stateLimit=stateLimit,
    epsilon=initialE,
    learningRate=lr,
    epsilonDecay=epsilonDecay,
    minEpsilon=minEpsilon,
    gamma=gamma,
    overwrite=overwrite
)

snakeTable.train(agentPlayOrder=['snake'], episodes=episodes)

# Next, get the best possible play through...
bestMemory = snakeTable.playGame(['snake'], random=False)
snakeTable.updateGameMetrics(bestMemory)
print(f'Best Game after {episodes} playthroughs: {snakeTable.snakeScores[-1]}')
