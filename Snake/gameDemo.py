"""
File: gameDemo.py
Location: /Snake/
Creation Date: 2020-09-24

The sole purpose is to house a test of the game.
Running this file lets the user play through a
game of snake, albeit crudely. The board will be
printed at each frame, and the user must input
a turn 'F' (forward), 'R' (right), or 'L' (left).
The rules are usual snake.
"""


from .SnakeEnv import SnakeGame
from .SnakeAgent import SnakeAgent
from .utils import *
import sys
import pprint

agent = SnakeAgent()
env = SnakeGame(agent, boardSize=15)
# Simulate the game, asking for
# direction in each frame...
allSnakeStates = []
while True:
    direction = input('Enter forward (F), left (L), right (R) or quit (Q): ')
    if direction != '':
        if direction == 'Q':
            print('Exporting to GIF...')
            exportGIF('game.gif', allSnakeStates, scale=20)
            print('Your score:', agent.score)
            sys.exit(1)
        state = env.stepForward(direction)
        allSnakeStates.append(state)
        if gameOver:
            print('Exporting to GIF...')
            env.exportGIF('game.gif', agent.snakeFrames, scale=20)
            print('Getting game memory in states...')
            codedStates = agent.getGameMemory()
            pprint.pprint(codedStates)
            print('Game over you died!')
            print('Your score:', agent.score)
            sys.exit(1)
