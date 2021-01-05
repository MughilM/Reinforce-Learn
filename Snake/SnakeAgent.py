"""
File: SnakeAgent.py
Location: /Snake/
Creation Date: 2020-09-19

This file implements the snake agent
(Hello, this is Snake) for the snake game. This
has the methods which actually play the game.
The actual snake mechanices are implemented
here, while the environment simply houses the
board and the food.
"""
import copy


class SnakeAgent:
    def __init__(self):
        self.currentFrame = []
        self.score = 0
        self.actionList = ['F', 'L', 'R']
        # Direction is from the snake's
        # perspective. Direction itself
        # is up, down, left, right. But
        # snake's turning is from the
        # snake's perspective e.g.
        # moving down and turning left
        # means the snake is now going RIGHT.
        self.DIR_RESULT = {
            'U': {
                'F': 'U',
                'L': 'L',
                'R': 'R'
            },
            'D': {
                'F': 'D',
                'L': 'R',
                'R': 'L'
            },
            'L': {
                'F': 'L',
                'L': 'D',
                'R': 'U'
            },
            'R': {
                'F': 'R',
                'L': 'U',
                'R': 'D'
            }
        }
        self.direction = ''
        self.gameOver = False
        self.reset()

    def reset(self):
        # Clean the previous states,
        # and put the snake in the top corner...
        # actionsRewards is one behind, because
        # it needs to match up against the previous
        # state, since it's storing the RESULT of
        # turning from that state.
        self.currentFrame = [(0, 0), (0, 1), (0, 2)]
        self.score = 3
        self.direction = 'R'
        self.gameOver = False

    def makeMove(self, turn, env):
        """
        The meat method. Given a turn ('F', 'L', 'R'),
        moves the snake in that direction by one
        step and returns appropriate reward. A reward of 1 is
        given if it eats a fruit, -1 if the snake dies, and
        0 otherwise. The locations are all updated automatically.
        If the move leads to a fruit, it eats the fruit,
        the snake gets longer,
        :param turn: One of 'F', 'L', or 'R'
        :param env: An instance of SnakeGame
        :return: The reward and whether it was a game over...
        """
        if turn not in self.actionList:
            raise ValueError(f'Action "{turn}" not in the action list!')
        if self.gameOver:
            print('Game is over! Please reset!')
            return
        newState = copy.deepcopy(self.currentFrame)
        newDirection = self.DIR_RESULT[self.direction][turn]
        for i in range(self.score - 1):
            newState[i] = newState[i + 1]
        # If we didn't change direction, push everything one...
        if newDirection == self.direction:
            # Using the current direction, update the head...
            if self.direction == 'U':
                newState[-1] = (newState[-1][0] - 1, newState[-1][1])
            elif self.direction == 'D':
                newState[-1] = (newState[-1][0] + 1, newState[-1][1])
            elif self.direction == 'L':
                newState[-1] = (newState[-1][0], newState[-1][1] - 1)
            else:
                newState[-1] = (newState[-1][0], newState[-1][1] + 1)
        # Changed direction.
        else:
            if newDirection == 'U':
                newState[-1] = (newState[-2][0] - 1, newState[-2][1])
            elif newDirection == 'D':
                newState[-1] = (newState[-2][0] + 1, newState[-2][1])
            elif newDirection == 'L':
                newState[-1] = (newState[-2][0], newState[-2][1] - 1)
            else:
                newState[-1] = (newState[-2][0], newState[-2][1] + 1)
        # Check to see if we've crashed...
        # Either we ate ourself or went out of bounds.
        if (newState[-1] in newState[:-1]) or \
                (any(r < 0 or c < 0 or r >= env.boardSize or c >= env.boardSize for r, c in newState)):
            self.gameOver = True
            reward = -10
        # Check to see if we've eaten a fruit.
        # Use the tail location of the previous
        # state to extend. Takes care of weird edge cases.
        elif newState[-1] == env.fruitLoc:
            newState.insert(0, self.currentFrame[0])
            self.score += 1
            reward = 10
        else:
            reward = 0  # We didn't crash or eat, so no reward
        self.direction = newDirection  # Set to new direction...
        self.currentFrame = newState
        return reward, self.gameOver

    def encodeCurrentState(self):
        """
        Primarily internal method. It will take the current
        state of the snake, and encode it according to our rules.
        It will use the most recent location of the snake in
        the environment variables.
        :return: The state coded as an 11-bit string:
            - Is there immediate danger in front, left,
            or right of the snake?
            - The direction of the fruit (up, down, left,
            right). From a top-down perspective. More
            than one is possible.
            - The direction of the snake (up, down, left,
            right)
        Thus, the coding is [danger ==> 'FLR']
        [fruit direction ==> 'UDLR']
        [snake direction ==> 'UDLR'] (mutually exclusive)
        """
        currState = self.currentFrame[-1]
        directionCode = {
            'U': '1000',
            'D': '0100',
            'L': '0010',
            'R': '0001'
        }
        coding = ''
        # For immediate danger, we look at the snake head, and see
        # if either the edge of the board or a snake body part is
        # next to it. The array is in FLR order.
        head = currState[-1]
        if self.direction == 'U':
            proximity = [
                (head[0] - 1, head[1]),
                (head[0], head[1] - 1),
                (head[0], head[1] + 1)
            ]
        elif self.direction == 'D':
            proximity = [
                (head[0] + 1, head[1]),
                (head[0], head[1] + 1),
                (head[0], head[1] - 1)
            ]
        elif self.direction == 'L':
            proximity = [
                (head[0], head[1] - 1),
                (head[0] + 1, head[1]),
                (head[0] - 1, head[1])
            ]
        else:
            proximity = [
                (head[0], head[1] + 1),
                (head[0] - 1, head[1]),
                (head[0] + 1, head[1])
            ]
        # Lotta stuff going on here:
        #   Check if each location is in the snake body
        #   or off the board. Convert the Trues and Falses
        # into a bit string we can directly attach to our coding.
        dangers = ((r, c) in currState or not (0 <= r < self.env.boardSize and 0 <= c < self.env.boardSize)
                   for r, c in proximity)
        coding += ''.join(map(lambda x: str(int(x)), dangers))

        # Now the fruit location. The fruit can't be both above and
        # below the snake, so append in pairs...
        fruitR, fruitC = self.env.fruitLocs[-1]
        if head[0] > fruitR:
            coding += '10'
        elif head[0] < fruitR:
            coding += '01'
        else:
            coding += '00'  # The fruit is on the same row
        # Left/right
        if head[1] > fruitC:
            coding += '10'
        elif head[1] < fruitC:
            coding += '01'
        else:
            coding += '00'

        # Now the direction of the snake...Straightforward
        coding += directionCode[self.direction]
        return coding

    def getGameMemory(self):
        """
        Converts all the frames of the game into a format
        for Q-learning. Ideally should be called when
        the game is over, but since we are saving
        the encoded states as we go along, you can
        call it whenever....
        :return: A list of tuples in the form
        (state, action, reward, nextState, gameOver).
        """
        # The length of the actionRewards list will always be
        # one less than the length of the states. Thus,
        # because of what we're keeping track of,
        # this is a simple loop...
        memory = []
        for i in range(len(self.actionsRewards)):
            currentState = self.encodedStates[i]
            nextState = self.encodedStates[i + 1]
            _, action, reward, gameOver = self.actionsRewards[i]
            memory.append((currentState, action, reward, nextState, gameOver))
        return memory
