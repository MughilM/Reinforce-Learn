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

from agents.discreteAgent import DiscreteAgent
from environment import Environment
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

        # If the path exists and overwrite is True, then erase
        # the contents....
        if os.path.exists(os.path.join(outputDir, experimentName)):
            if overwrite:
                shutil.rmtree(os.path.exists(os.path.join(outputDir, experimentName)))
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

    def saveDataArtifcats(self, **kwargs):
        """
        This method is the save version of `loadDataArtifacts`. If there is anything else
        other than the Q tables that need to be saved (such as plots or gameplays), this
        method should be implemented.
        :return:
        """
        raise NotImplementedError('Saving of extra data not implemented!')

    def playGame(self, firstTurn):
        """
        Plays a game according to the steps defined in the
        environment's stepForward. This method is designed to be
        environment-agnostic, meaning changing the rules will change
        the behaviour. This will also return the encoded game
        memory.
        :return: A list of [state, agentName, action, reward, nextState, gameOver]
        for however long the game lasts...
        """
        self.gamesPlayed += 1
        # Keep a counter for number of states, to prevent
        # continuous looping in a game...
        stateCounts = 1
        # Reset the environment...
        currentState = self.env.reset()
        # Set the agent to move first, a list
        # of all agents, and how many agents there
        # so we can loop through the list each time...
        # TODO: Provide agent turn order...
        allAgents = list(self.env.agents.keys())
        agentToPlay = firstTurn
        agentIndex = allAgents.index(agentToPlay)
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
            if np.random.rand() < self.epsilon:
                action = self.env.agents[agentToPlay].chooseRandomAction()
            else:
                mappedRow = self.mapStateToRow(currentEncodedState)
                rowData = self.QTables[agentToPlay][mappedRow]
                action = self.actionList[np.argmax(rowData)]
            # Step forward!
            currentState, reward, gameOver = self.env.stepForward(agentToPlay, action)
            # Append the state/action/reward/gameOver
            # Put a placeholder for the next state. It gets filled in at the start of the next loop)
            gameMemory.append([currentEncodedState, agentToPlay, action, reward, gameOver])
            stateCounts += 1
            # It is now the next agent's turn. Increment the index,
            # but also mod the number of agents we have, for automatic looping...
            agentIndex = (agentIndex + 1) % numOfAgents
            agentToPlay = allAgents[agentIndex]
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
        self.epsilon = max(self.epsilon * self.epsilon, self.minEpsilon)




