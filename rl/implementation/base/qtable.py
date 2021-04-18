"""
File: qtable.py
Creation Date: 2021-02-13
Location: base

This file contains the base implementation of a QTable.
By default, each agent has its own table. Two things
that must be provided are the shapse of the tables and
a way to map the encoded state returned by the environment
to the row of the table. The goal of this class is for
future games that use the QTables to provide as few
information as possible.
"""

import numpy as np

from .agents.discreteAgent import DiscreteAgent
from .environment import Environment
from typing import List, Dict
import os
import sys
import shutil


class QTable:
    def __init__(self, outputDir, experimentName, rows, cols, discreteAgents: Dict[str, DiscreteAgent],
                 environment: Environment, stateLimit=10000,
                 epsilon=1, learningRate=0.1, epsilonDecay=0.995,
                 minEpsilon=0.01, gamma=0.95, overwrite=False):
        self.gamesPlayed = 0
        self.maxScore = 0
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.epsilonDecay = epsilonDecay
        self.minEpsilon = minEpsilon
        self.gamma = gamma
        self.stateLimit = stateLimit
        # Save the discrete action list...
        self.actionList = discreteAgents[list(discreteAgents.keys())[0]].actionList

        # Set up a QTable for each agent...
        self.QTables = {agentName: np.zeros((rows, cols), dtype=float) for agentName in discreteAgents.keys()}
        self.env = environment

        # Finally, any time we save Q-tables and other things...
        self.outputDir = outputDir
        self.expName = experimentName
        self.overwrite = overwrite

        # Flag to see if we have notified the user of
        # reaching minimum epsilon...
        self.notified = False

        # If the path exists and overwrite is True, then erase
        # the contents....
        if os.path.exists(os.path.join(outputDir, experimentName)):
            if overwrite:
                shutil.rmtree(os.path.join(outputDir, experimentName))
            else:
                print('WARNING: Experiment directory exists and overwriting is disabled! '
                      'Please rerun with overwriting enabled or provide another experiment'
                      'name.')
                sys.exit(1)
        # Create the directory...
        os.makedirs(os.path.join(outputDir, experimentName))

    def mapStateToRow(self, encodedState):
        """
        This function allows for correct updating of the tables.
        Given a state, return a corresponding row in the table. It is
        up to the user to make sure there are no conflicts.
        :param encodedState:
        :return: The row of the table.
        """
        raise NotImplementedError('Please provide a mapping from the state to the row!')

    def loadQTables(self):
        """
        This method loads the Q-tables that were saved using the
        `saveQTables` function. They are stored in npz format.
        :return:
        """
        self.QTables = np.load(os.path.join(self.outputDir, self.expName, 'tables.npz'))

    def loadDataArtifacts(self, **kwargs):
        """
        This is a user-defined method. Other than the Q tables, if there is anything
        else that needs to be loaded, then this method should be implemented. This
        method isn't used by the QTable class, and should be used separately.
        :return:
        """
        raise NotImplementedError('Extra loading of data not implemented yet!')

    def saveQTables(self):
        """
        This method saves the Q-tables in .npz format in the directory defined by the
        experiment name.
        :return:
        """
        # Unpack the Q table dictionary into the keyword arguments. That way,
        # when the npz is saved, the keys become the agent names...
        np.savez(os.path.join(self.outputDir, self.expName, 'tables.npz'), **self.QTables)

    def saveDataArtifacts(self, **kwargs):
        """
        This method is the save version of `loadDataArtifacts`. If there is anything else
        other than the Q tables that need to be saved (such as plots or gameplays), this
        method should be implemented. Of course, extra variables needed to do this saving
        should be added in the subclass, not here.
        :return:
        """
        raise NotImplementedError('Saving of extra data not implemented!')

    # TODO: Eventually, provide a way for agents to go through
    # play that doesn't involve one turn after the other i.e.
    # the next player's turn is determined by what happened on
    # the previous turn, or something like that...
    def playGame(self, agentPlayOrder: List[str], random=True):
        """
        Plays a game according to the steps defined in the
        environment's stepForward. This method is designed to be
        environment-agnostic, meaning changing the rules will change
        the behaviour. This will also return the encoded game
        memory.
        :param: agentPlayOrder: A list of the names of agents that
        play in designatad order e.g. input of ['a1', 'a2', 'a3'] means
        'a1' play first, 'a2' plays second, and 'a3' plays third,
        then it is 'a1''s turn again.
        :param: random: A flag to say if the play should include
        random choices. If set to False, then this will use the MAX
        Q VALUES that are currently saved. Useful for checking the best
        possible play through after a long training session...Default True..
        :return: A list of [state, agentName, action, reward, nextState, gameOver]
        for however long the game lasts...
        """
        self.gamesPlayed += 1
        # Keep a counter for number of states, to prevent
        # continuous looping in a game...
        stateCounts = 1
        # Reset the environment...
        _ = self.env.reset()
        # Set the agent to move first, a list
        # of all agents, and how many agents there
        # so we can loop through the list each time...
        agentIndex = 0
        agentToPlay = agentPlayOrder[agentIndex]
        numOfAgents = self.env.getNumberOfAgents()
        gameOver = False
        gameMemory: List[List] = []
        while not gameOver and stateCounts < self.stateLimit:
            currentEncodedState = self.env.encodeCurrentState()
            # If it's at least the second state, then it was the "next state"
            # of the previous state/action...
            if len(gameMemory) > 0:
                gameMemory[-1][4] = currentEncodedState
            # With an epsilon % chance, choose a random action.
            # Otherwise, choose the action with the largest Q-value..
            if random and np.random.rand() < self.epsilon:
                action = self.env.agents[agentToPlay].chooseRandomAction()
            else:
                mappedRow = self.mapStateToRow(currentEncodedState)
                rowData = self.QTables[agentToPlay][mappedRow]
                action = self.actionList[np.argmax(rowData)]
            # Step forward!
            currentState, reward, gameOver = self.env.stepForward(agentToPlay, action)
            # Append the state/action/reward/gameOver
            # Put a placeholder for the next state. It gets filled in at the start of the next loop)
            gameMemory.append([currentEncodedState, agentToPlay, action, reward, None, gameOver])
            stateCounts += 1
            # It is now the next agent's turn. Increment the index,
            # but also mod the number of agents we have, for automatic looping...
            agentIndex = (agentIndex + 1) % numOfAgents
            agentToPlay = agentPlayOrder[agentIndex]
        # Game is over, so add the last state to the game memory and return...
        gameMemory[-1][4] = self.env.encodeCurrentState()
        return gameMemory

    def updateTable(self, gameMemory):
        """
        Given a game memory returned from playing a game, this will update the Q-table according
        to Bellman's equation.
        :param gameMemory: A single game's memory. Assumed to end in a game over...
        :return:
        """
        for memory in gameMemory:
            currState, agentName, action, reward, nextState, gameOver = tuple(memory)
            row = self.mapStateToRow(currState)
            col = self.actionList.index(action)
            # If it's a game over, there is no maximum Q-value of the next state.
            # However, we initialized the Q table with zeroes,
            # so the line would still check out.
            nextRow = self.QTables[agentName][self.mapStateToRow(nextState)]
            maxNextQValue = max(nextRow)
            # Update, Q(s, a) = Q(s, a) + alpha * ( r(s, a) + gamma * maxNextQValue - Q(s,a) )...
            self.QTables[agentName][row, col] += self.learningRate * (reward + self.gamma * maxNextQValue -
                                                                      self.QTables[agentName][row, col])
        # Decay the equation for the next game...
        self.epsilon = max(self.epsilon * self.epsilonDecay, self.minEpsilon)
        if not self.notified and self.epsilon == self.minEpsilon:
            print('\nWARNING: Minimum epsilon reached! If this is well before expected,'
                  ' consider lowering the decay for next training session...')
            self.notified = True

    def updateGameMetrics(self, gameMemory):
        """
        Given the memory of the game just finished (as well as all the class variables), this
        method will save any extra metrics that are to be defined by the user. Maybe these
        metrics will be used by `load` and `saveDataArtifacts`. Regardless, this needs
        to be implemented.
        :param gameMemory:
        :return:
        """
        raise NotImplementedError('Please update custom game metrics!')

    def train(self, agentPlayOrder, episodes):
        """
        The method which allows us to train and update the Q table. This will
        use the variables which are saved in the object, things like learning rate,
        initial epsilon, epsilon decay rate, Each time it plays, a memory is
        produced, and this memory is used to update the Q table and update
        the game metrics. When all the episodes have been finished off,
        the Q tables and data artifacts are saved to their directories...
        :param: agentPlayOrder: The play order for the agents. Should be
        list of agent names on who goes first, second, etc...
        :param: episodes: The number of episodes to train for i.e. the number
        of times we play the game.
        :return: Nothing.
        """
        # As the doc says, we are assuming all the variables have
        # been set up already. Thus, we just run the loop...
        for _ in range(episodes):
            memory = self.playGame(agentPlayOrder)
            self.updateTable(memory)
            self.updateGameMetrics(memory)
            print(f'\rGame {self.gamesPlayed} / {episodes}', end='')
        # We are done, save the tables and other data artifacts...
        self.saveQTables()
        self.saveDataArtifacts()
        print()




