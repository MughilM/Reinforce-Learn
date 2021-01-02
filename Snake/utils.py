"""
File: utils.py
Location: /Snake/
Creation Date: 2021-01-01

This file houses any utility functions that is filled as needed.
Most notable are those functions which can make the debugging
process easier. Other functions that can't exactly fall into
any other category can also be placed here. For example, when
I created this file, the intention was to place the code to generate
the GIFs for when a game is played.
"""
import numpy as np
from itertools import product


def produceBoardFrame(snakeGameState: dict, scale=1):
    """
    This method produces ONE frame in image format of
    the current game state, including the snake and fruit.
    The starting length of the snake is needed to compute
    the correct fruit location.
    :param snakeGameState: A dictionary object that is
    the output of the stepForward() function. The
    expected keys are `boardSize`, `snakeLocs`, and
    'fruitLoc`
    :param scale: How much to blow up the image. Scale
    of 1 means that each location index takes up just
    one pixel.
    :return: A grayscale-valued ndarray depicting the
    current state of the entire snake environment.
    """
    # First check for the correct keys...
    if any(key not in snakeGameState.keys() for key in ['boardSize', 'snakeLocs', 'fruitLoc']):
        raise ValueError('The given snake state does not have the correct keys!')
    boardSize = snakeGameState['boardSize']
    frame = snakeGameState['snakeLocs']
    fruitR, fruitC = snakeGameState['fruitLoc']
    gameFrame = np.zeros(shape=((boardSize + 2) * scale, (boardSize + 2) * scale),
                         dtype=np.uint8)
    # Put the border...
    gameFrame[:scale, :] = 50
    gameFrame[-scale:, :] = 50
    gameFrame[:, :scale] = 50
    gameFrame[:, -scale:] = 50
    # Upscale the frame indices so that each original coordinate
    # will now represent the upper left corner of a "box".
    offsetSnakeBody = [((r + 1) * scale, (c + 1) * scale) for r, c in frame[:-1]]
    # Now add the rest of each box...
    fullBodyLocs = []
    for r, c in offsetSnakeBody:
        fullBodyLocs.extend(product(range(r, r + scale), range(c, c + scale)))
    # To easily assign values to the gameFrame, a change
    # of format is necessary...
    formattedBody = tuple(zip(*[(r, c) for r, c in fullBodyLocs]))
    # The head and the fruit are different colors, so
    # format those the same way...
    formattedFruitLoc = tuple(zip(*[(r, c) for r, c in
                                    product(range((fruitR + 1) * scale, (fruitR + 2) * scale),
                                            range((fruitC + 1) * scale, (fruitC + 2) * scale))]))
    headR, headC = frame[-1]
    formattedHeadLoc = tuple(zip(*[(r, c) for r, c in
                                   product(range((headR + 1) * scale, (headR + 2) * scale),
                                           range((headC + 1) * scale, (headC + 2) * scale))]))
    # The body is white, the head is slightly
    # darker, the fruit is even more darker...
    gameFrame[formattedBody] = 255
    gameFrame[formattedHeadLoc] = 220
    gameFrame[formattedFruitLoc] = 128

    return gameFrame

