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
import sys
import pprint

env = SnakeGame()
agent = SnakeAgent(environment=env)
# Simulate the game, asking for
# direction in each frame...
while True:
    print('Snake Locations:', agent.snakeFrames[-1])
    print(env.boardToString(agent.snakeFrames[-1]))
    print('Current state:', agent.encodeCurrentState())
    direction = input('Enter forward (F), left (L), right (R) or quit (Q): ')
    if direction != '':
        if direction == 'Q':
            env.exportGIF('game.gif', agent.snakeFrames, scale=20)
            print(f'Encoded states\n==========\n{agent.encodedStates}')
            print(f'Actions and rewards\n============\n{agent.actionsRewards}')
            print(f'Final Game Memory\n=============\n{pprint.pformat(agent.getGameMemory())}')
            print('Your score:', agent.score)
            sys.exit(1)
        reward, gameOver = agent.makeMove(direction)
        if gameOver:
            print('Exporting to GIF...')
            env.exportGIF('game.gif', agent.snakeFrames, scale=20)
            print('Getting game memory in states...')
            codedStates = agent.getGameMemory()
            pprint.pprint(codedStates)
            print('Game over you died!')
            print('Your score:', agent.score)
            sys.exit(1)
