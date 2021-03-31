"""
File: main.py
Location: /rl/implementation
Creation Date: 2021-02-20

Whenever an experiment needs to be executed, this file is run. In this file, the user
can provide custom run functions for each method. I thought about the best way to
put an umbrella over all possible ways someone might want to run, and I eventually
just settled on a custom run function for each game. The reason is that there too much
customization in how games are setup that it's almost impossible to create a single
function to run them all. Mostly due to custom parameters in the environment and agent
that must be provided, and custom ways that possible multi-agents must be setup.
All that needs to be done is to map the argument to the function.
"""
from .config import *
from .envs.snake.game import *
import argparse
import os

FUNCTIONS = {
    'snake11': runSnake11
}

def runSnake11(expName, boardSize, episodes, lr, initialE, minE, epsilonDecay, gamma, overwrite):
    # Create the environment with the specified board size
    # and create the singular agent...
    agentName = 'snake'
    snakeAgent = SnakeAgent(actionList=['F', 'L', 'R'])
    snakeEnv = SnakeEnv({agentName: snakeAgent}, boardSize=boardSize)

    # Create the Q table using the rest of the variables..
    snakeTable = SnakeQTable(
        outputDir=EXP_OUTPUT_DIR,
        experimentName=expName,
        rows=2 ** 11,
        cols=3,
        discreteAgents={agentName: snakeAgent},
        environment=snakeEnv,
        stateLimit=5000,
        epsilon=initialE,
        learningRate=lr,
        epsilonDecay=epsilonDecay,
        minEpsilon=minE,
        gamma=gamma,
        overwrite=overwrite
    )
    # The Qtable has its own train method. So just run that...
    snakeTable.train(agentPlayOrder=[agentName], episodes=episodes)

    # Play through the best possible game
    bestmemory = snakeTable.playGame([agentName], random=False)
    snakeTable.updateGameMetrics(bestMemory)
    print(f'Best game after {episodes} play throughs: {snakeTable.snakeScores[-1]}')



def run(game, expName, boardSize, episodes, lr, initialE, minE, epsilonDecay, gamma, overwrite):
    """
    The main worker method. This will have all the options necessary so that
    one of the existing reinforcement learning implementations is either
    trained or played with. The agents could be games or some other thing.
    The options and the in-built dictionary allow the correct classes
    to be called based on the arguments (see below.....
    :return:
    """
    # Each game is defined in the dictionary. To run it properly,
    # we need the environment object, agent object, and the Q table class.
    # We need to accomodate the possibility of including environment parameters,
    # but maybe we'll deal with that later TODO.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program to train/play with reinforcement learning agents.')
    # Add all the arguments (there are a lot of them...)
    parser.add_argument('game', choices=['snake'],
                        help='The type of game to play/train. Current choices are "snake".')
    parser.add_argument('expName', required=True,
                        help='The experiment name. All data artifacts that might get saved will do so under'
                             'this subfolder in the EXP_OUTPUT_DIR defined in config.py. Currently, it'
                             f'is {EXP_OUTPUT_DIR}.')
    parser.add_argument('--board-size', type=int, default=10,
                        help='The board size for snake. Only gets used if snake is selected as the'
                             'game. Default is a 10 by 10 grid.')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of times to play the game, for training. Default 100.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate for the RL algorithm. Default 0.001.')
    parser.add_argument('--initial-e', type=float, default=1.0,
                        help='The initial value of epsilon for our exploration/exploitation. Default 1.')
    parser.add_argument('--min-e', type=float, default=0.01,
                        help='The minimum value of epsilon i.e. the minimum chance to do something random.'
                             ' Default 1%')
    parser.add_argument('--e-decay', type=float, default=0.995,
                        help='The percentage decay of epsilon every episode. Default 99.5%.')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='The gamma value for RL i.e. how much decay to have for past actions. Default 0.99.')
    parser.add_argument('--overwrite', type=bool, default=False,
                        help='Whether to overwrite the folder where Q-table and model artifacts get outputted to.'
                             'Default False i.e. DO NOT OVERWRITE current directories.')

    # Parse all the arguments!
    args = parser.parse_args()

    # Make the output directories corresponding to the experiment name...
    os.makedirs(os.path.join(BASE_DIR, EXP_OUTPUT_DIR, args.expName), exist_ok=True)


