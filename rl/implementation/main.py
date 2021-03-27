"""
File: main.py
Location: /rl/implementation
Creation Date: 2021-02-20

Whenever an experiment needs to be executed, this file is run. In this file, the user
can provide which environment to run the experiment in. In addition,
the user can provide arguments to fine-tune the experiment, such as whether s/he wants to
run a Q-Table or a neural network implementation. Various hyperparameters such as number of games,
gamma, etc. are also available. The corresponding file from /envs/ is grabbed and ran.
"""
from .config import *
from .envs.snake.game import *
import argparse
import os

GAMES = {
    'snake': SnakeQTable
}

def run(game, expName, boardSize, episodes, lr, initialE, minE, epsilonDecay, gamma, overwrite):
    """
    The main worker method. This will have all the options necessary so that
    one of the existing reinforcement learning implementations is either
    trained or played with. The agents could be games or some other thing.
    The options and the in-built dictionary allow the correct classes
    to be called based on the arguments (see below.....
    :return:
    """
    # First grab the gameObject defined in our directory...
    gameObj = GAMES[game]
    # This one is guaranteed to be of type QTable... which means
    # all the extra arguments, like the epsilon related stuff, gamma, etc.
    # can be passed directly into the object...
    # We also make to deal with extra configuration variables (such as board size for Snake)

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


