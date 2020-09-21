######################################################
# File: playSnake.py
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
        os.makedirs('./Data/imgs/', exist_ok=True)
        os.makedirs('./Data/gifs/', exist_ok=True)
        # Force a minimum size of 5...
        if boardSize < 5:
            raise ValueError("Board size of {} is too small!".format(boardSize))
        self.boardSize = boardSize
        # Don't pay attention to the values here.
        # They'll get reset. It's just so my IDE can
        # recognize the data types used :)
        self.fruitLoc = (0, 0)
        self.reset()

    def reset(self):
        """
        Resets the board and places the snake
        in the top corner. It will always
        be of length 3. Randomly places the
        fruit as well.
        :return: Starting position of snake
        """
        fruitCellNum = np.random.randint(low=3, high=self.boardSize ** 2)
        self.fruitLoc = (fruitCellNum // self.boardSize, fruitCellNum % self.boardSize)
        # Return top 3 corner cells...
        return [(0, 0), (0, 1), (0, 2)]

    def exportGIF(self, filename: str, frames: list, fruitLocs: list):
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
        :param: frutiLocs: A list containing the fruit
        locations at each frame. Should be the same length
        as `frames`.
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
        playedGame = np.zeros(shape=(len(frames), self.boardSize + 2, self.boardSize + 2), dtype=float)
        # Put the border...
        playedGame[:, 0, :] = 1
        playedGame[:, -1, :] = 1
        playedGame[:, :, 0] = 1
        playedGame[:, :, -1] = 1
        filename = os.path.join('./Data/', filename)
        with imageio.get_writer(filename, mode='I') as writer:
            for i, (frame, fruit) in enumerate(zip(frames, fruitLocs)):
                # The format is the reverse of zip
                correctedSnakeLocs = zip(*[(i, r + 1, c + 1) for r, c in frame])
                correctedFruitLoc = (i, fruit[0] + 1, fruit[1] + 1)
                playedGame[correctedSnakeLocs] = 1
                playedGame[correctedFruitLoc] = 0.3
                # Color the head...
                playedGame[i, frame[-1][0] + 1, frame[-1][1] + 1] = 0.7

                # Append to the GIF
                writer.append_data(playedGame[i])
