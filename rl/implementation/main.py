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

It's important that each custom run function has the variables in the same order,
as these get called from main. Any extra keyword arguments for things like
envirnoment parameters and the like should be in the end, as these get
passed through kwargs at the end.
"""
from .config import *
from .envs.snake.game import *
import argparse
import os

# List of arguments to treat as default. These will have
# their own parameter name in the run function. Any variable not listed here
# will get passed through kwargs.
DEFAULT_ARGS = {'expName', 'episodes', 'lr', 'initialE', 'minE',
                'epsilonDecay', 'gamma', 'overwrite'}

def runSnake11(expName, episodes, lr, initialE, minE, epsilonDecay, gamma, overwrite, **kwargs):
    # The board size is needed. Get it from kwargs...
    boardSize = kwargs['boardSize']
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
    bestMemory = snakeTable.playGame([agentName], random=False)
    snakeTable.updateGameMetrics(bestMemory)
    print(f'Best game after {episodes} play throughs: {snakeTable.snakeScores[-1]}')


FUNCTIONS = {
    'snake11': runSnake11
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program to train/play with reinforcement learning agents.')
    # Add all the arguments (there are a lot of them...)
    parser.add_argument('game', choices=['snake11'],
                        help='The type of game to play/train. Current choices are "snake".')
    parser.add_argument('expName',
                        help='The experiment name. All data artifacts that might get saved will do so under'
                             'this subfolder in the EXP_OUTPUT_DIR defined in config.py. Currently, it'
                             f'is {EXP_OUTPUT_DIR}.')
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
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Whether to overwrite the folder where Q-table and model artifacts get outputted to.'
                             'Default False i.e. DO NOT OVERWRITE current directories.')

    # Custom arguments needed per game...
    parser.add_argument('--board-size', type=int, default=10,
                        help='The board size for snake. Only gets used if snake is selected as the'
                             'game. Default is a 10 by 10 grid.')

    # Parse all the arguments!
    args = parser.parse_args()

    # Make the output directories corresponding to the experiment name...
    os.makedirs(os.path.join(BASE_DIR, EXP_OUTPUT_DIR, args.expName), exist_ok=True)

    # Get the corresponding run function from the dictionary
    runFunc = FUNCTIONS[args.game]

    runFunc(expName=args.expName, episodes=args.episodes, lr=args.lr, initialE=args.initial_e,
            minE=args.min_e, epsilonDecay=args.e_decay, gamma=args.gamma, overwrite=args.overwrite,
            **{'boardSize': args.board_size})



