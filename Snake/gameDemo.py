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
env = SnakeGame(agent, boardSize=10)
# Simulate the game, asking for
# direction in each frame...
allSnakeStates = []
state = env.reset()
while True:
    print(boardToString(state))
    direction = input('Enter forward (F), left (L), right (R) or quit (Q): ')
    if direction != '':
        if direction == 'Q':
            print('Exporting to GIF...')
            exportGIF(allSnakeStates, 'game.gif', scale=20)
            print('Your score:', agent.score)
            sys.exit(1)
        state, reward, gameOver = env.stepForward(direction)
        allSnakeStates.append(state)
        print(boardToString(state))
        if gameOver:
            print('Game over you died!')
            print('Your score:', agent.score)
            print('Exporting to GIF...')
            exportGIF(allSnakeStates, 'game.gif', scale=20)
            sys.exit(1)
