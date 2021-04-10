"""
File: BaseDQN.py
Creation Date: 2021-04-01
Location: implementation/base/DQN

A barebones DQN class. Has everything to train a bare model. This is similar to the Qtable
in that this also has a play game function. The reason they are separate, is because of
the way this plays the game is different, since its prediction method for the next action
is neural network based instead of table based.
"""

import tensorflow as tf
from tensorflow.keras.models import *
from typing import *
from ..agents.basicAgent import Agent
from ..environment import Environment


class BaseDQN:
    def __init__(self, outputDir, experimentName, agents: Dict[str, Agent], environment: Environment,
                 stateLimit=10000, epsilon=1, learningRate=0.1, epsilonDecay=0.995,
                 minEpsilon=0.01, gamma=0.95, overwrite=False):
        """
        The base initializer. Includes many default values for training. Anything else should be
        included in sub classes.
        :param outputDir: The directory to output all model artifacts, things like checkpoints and the like
        :param experimentName: The name of the experiment. Model artifacts are saved in subfolder of
        the same name
        :param agents: A dictionary of agents. Could be more than one, and turn order is enforced through
        agent naems.
        :param environment: The environment object. Maybe needed to access environment variables.
        :param stateLimit: The max state limit to stop the current game. Prevents infinite looping between
        the same set of states.
        :param epsilon: The randomness parameter. A random action is chosen with this probability.
        :param learningRate: Learning rate of model.
        :param epsilonDecay: Epsilon decreases by this percentage each episode.
        :param minEpsilon: The minimum possible value of epsilon.
        :param gamma: "Lookback" parameter.
        :param overwrite: Whether to overwrite the experiment folder with results.
        """
        self.gamesPlayed = 0
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.epsilonDecay = epsilonDecay
        self.minEpsilon = minEpsilon
        self.gamma = gamma
        self.stateLimit = stateLimit

        # Make the target model from createModel...
        self.targetModel = self.createModel()

        # Folder locations...
        self.outputDir = outputDir
        self.expName = experimentName
        self.overwrite = overwrite

    def createModel(self) -> Model:
        """
        Method that defines the model to be used and returns it. Should be custom built for each
        RL implementation.
        :return: A Model object.
        """
        raise NotImplementedError('Please implement the createModel function so we can train :)')

    def loadModel(self):
        """
        Simply loads the model from the experiment directory. This has been saved in a standard
        format. The structure of the model has also saved here, so information about structure
        is not needed.
        :return:
        """
        pass

    def loadDataArtifacts(self, **kwargs):
        """
        Same as the Qtable implementation. Anything extra that needs to be loaded outside of the model
        can be implemented here...Things like databases, etc.
        :param kwargs: Custom arguments...
        :return:
        """
        raise NotImplementedError('Please implement custom load data artifacts method. Allowed to put pass.')

    def saveModel(self):
        """
        Saves the entire model, so structure, optimizer state, and of course model weights are all saved.
        It is saved in the output directory under a subfolder of the experiment name.
        :return:
        """
        self.targetModel.save()

    def saveDataArtifacts(self, **kwargs):
        """
        Complementary to the loadDataArtifacts method. Anything that needs extra saving is here. Things
        like plots, csvs, etc.
        :param kwargs:
        :return:
        """

    def playGame(self):
        """
        How we play the game. This is different than the QTable function because now, our prediction

        :return:
        """