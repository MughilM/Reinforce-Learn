######################################################
# File: SnakeEnv.py
# Creation Date: 20/09/2020
# Location: /Snake/
#
# This file implements the simple environment
# for the snake game. It only has the game board's
# properties, because the snake can be considered an
# agent himself. Thus, for the environment, we only
# store the size of the board and the location of the
# fruit. It also provides capability to export the
# played game as a GIF, given the locations
# of the snake at each frame.
######################################################

import numpy as np
from itertools import product
import imageio
import os


class SnakeGame:
    def __init__(self, boardSize=10):
        """
        Saves the board size and randomly
        places the fruit. Same functionality
        as reset(), except the board size
        is saved. Board size larger than 5 is
        required.
        :param boardSize: Side length of board
        """
        # Create directories...
        os.makedirs('./Snake/Data/imgs/', exist_ok=True)
        os.makedirs('./Snake/Data/gifs/', exist_ok=True)
        # Force a minimum size of 5...
        if boardSize < 5:
            raise ValueError("Board size of {} is too small!".format(boardSize))
        self.boardSize = boardSize
        # Don't pay attention to the values here.
        # They'll get reset. It's just so my IDE can
        # recognize the data types used :)
        self.fruitLocs = []
        self.placedFruit = False
        self.reset()

    def reset(self):
        """
        Resets the board. Nothing
        to do here as the board itself
        doesn't change.
        :return:
        """
        self.placedFruit = False
        self.fruitLocs = []
        return

    def placeFruit(self, snakeLocs):
        """
        Places the fruit at random depending on the
        location of the snake. You should call this
        from the agent right after resetting and right
        after a fruit is eaten.
        :param snakeLocs:
        :return:
        """
        validLocs = [(r, c) for r, c in product(range(self.boardSize), repeat=2)
                     if (r, c) not in snakeLocs]
        # Randomly select one...
        selectionIndex = np.random.choice(len(validLocs))
        self.fruitLocs.append(validLocs[selectionIndex])
        self.placedFruit = True

    def exportGIF(self, filename: str, frames: list):
        """
        Takes the previous states and exports them
        as a playable GIF. Automatically places
        it in the ./Data/gifs/ folder.
        :param: filename: Name of GIF file. Automatically
        prepended with ./Data/ and adds .gif if not there
        :param: frames: A list containing the locations
        of the snake at each frame. Expects each element
        to be a list of two-tuples. The last element
        is the snake head.
        :return:
        """
        if not filename.endswith('.gif'):
            filename += '.gif'
        # We will convert these to numpy arrays
        # An ndarray of shape
        # (num of frames, board size + 2, board size + 2)
        # We add 2 to the size to show the border...
        # This also means we need to correct the indices.
        # Border and snake body will be white. Snake
        # head will half gray, and the fruit will be
        # a darker gray.
        playedGame = np.zeros(shape=(len(frames), self.boardSize + 2, self.boardSize + 2), dtype=np.uint8)
        # Put the border...
        playedGame[:, 0, :] = 50
        playedGame[:, -1, :] = 50
        playedGame[:, :, 0] = 50
        playedGame[:, :, -1] = 50
        filename = os.path.join('./Snake/Data/gifs', filename)
        # The fruit locations are only added every time the
        # snake eats, so we have to see when it eats, given
        # when the snake gets longer...
        with imageio.get_writer(filename, mode='I') as writer:
            fruitIndex = 0
            for i, frame in enumerate(frames):
                # The format is the reverse of zip
                correctedSnakeLocs = zip(*[(i, r + 1, c + 1) for r, c in frame])
                # print(list(correctedSnakeLocs))
                if i > 0 and len(frames[i]) > len(frames[i - 1]):
                    fruitIndex += 1
                correctedFruitLoc = (i, self.fruitLocs[fruitIndex][0] + 1, self.fruitLocs[fruitIndex][1] + 1)
                playedGame[tuple(correctedSnakeLocs)] = 255
                playedGame[correctedFruitLoc] = 128
                # Color the head a slightly off white...
                playedGame[i, frame[-1][0] + 1, frame[-1][1] + 1] = 220

                # Append to the GIF
                writer.append_data(playedGame[i])

    def boardToString(self, snakeLocs: list):
        """
        Given the locations of the snake, prints
        out the board.
        :param snakeLocs: The locations of the snake.
        :return:
        """
        gameStr = np.zeros(shape=(self.boardSize + 2, self.boardSize + 2), dtype=str)
        gameStr[:, :] = '-'
        # Hashtags for board border...
        gameStr[0, :] = '#'
        gameStr[-1, :] = '#'
        gameStr[:, 0] = '#'
        gameStr[:, -1] = '#'
        # 's' for snake body, 'h' for haed, 'f' for fruit
        gameStr[self.fruitLocs[-1][0] + 1, self.fruitLocs[-1][1] + 1] = 'f'
        for r, c in snakeLocs[:-1]:
            gameStr[r + 1, c + 1] = 's'
        gameStr[snakeLocs[-1][0] + 1, snakeLocs[-1][1] + 1] = 'h'
        return '\n'.join(''.join(row) for row in gameStr)
