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

def run():
    """
    The main worker method. This will have all the options necessary so that
    one of the existing reinforcement learning implementations is either
    trained or played with. The agents could be games or some other thing.
    The options and the in-built dictionary allow the correct classes
    to be called based on the arguments (see below)...
    :return:
    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program to train/play with reinforcement learning agents.')
    # Add all the arguments (there are a lot of them...)
    parser.add_argument('game', choices=['snake'])
    parser.add_argument()

