"""
File: BaseDQN.py
Creation Date: 2021-04-01
Location: implementation/base/DQN

A barebones DQN class. Has everything to train a bare model. This is similar to the Qtable
in that this also has a play game function. The reason they are separate, is because of
the way this plays the game is different, since its prediction method for the next action
is neural network based instead of table based.
"""


from tensorflow.keras.models import Model
from typing import Dict, List
from ..agents.basicAgent import Agent
from ..environment import Environment
from ..utils.experienceReplay import ExperienceReplay

import numpy as np
import os
import shutil
import sys


class BaseDQN:
    def __init__(self, outputDir, experimentName, agents: Dict[str, Agent], environment: Environment,
                 agentPlayOrder: List[str], stateLimit=10000, epsilon=1, learningRate=0.1, epsilonDecay=0.995,
                 minEpsilon=0.01, gamma=0.95, maxBufferSize=2000, overwrite=False):
        """
        The base initializer. Includes many default values for training. Anything else should be
        included in sub classes.
        :param outputDir: The directory to output all model artifacts, things like checkpoints and the like
        :param experimentName: The name of the experiment. Model artifacts are saved in subfolder of
        the same name
        :param agents: A dictionary of agents. Could be more than one, and turn order is enforced through
        agent names.
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
        self.maxBufferSize = maxBufferSize
        self.replay = ExperienceReplay(maxBufferSize)

        self.agentPlayOrder = agentPlayOrder
        self.turnIndex = 0
        self.agentToPlay = agentPlayOrder[self.turnIndex]

        # Make the target model from createModel...
        self.targetModel = self.createModel()

        # Folder locations...
        self.outputDir = outputDir
        self.expName = experimentName
        self.overwrite = overwrite

        # Environment and agents...
        self.env = environment
        self.agents = agents

        if os.path.exists(os.path.join(outputDir, experimentName)):
            if overwrite:
                shutil.rmtree(os.path.join(outputDir, experimentName))
            else:
                print('WARNING: Experiment directory exists and overwriting is disabled! '
                      'Please rerun with overwriting enabled or provide another experiment name.')
                sys.exit(1)
        # Create the directory
        os.makedirs(os.path.join(outputDir, experimentName))

    def createModel(self) -> Model:
        """
        Method that defines the model to be used and returns it. Should be custom built for each
        RL implementation.
        :return: A Model object.
        """
        raise NotImplementedError('Please implement the createModel function so we can train :)')

    def chooseActionFromPrediction(self, QValPreds):
        """
        This is an important method. Given the Q-value predictions that are popped out from
        the model, how are we going to use that to make an action decision? We don't know whether
        we have a continuous action space (in which case we need the output directly), or
        discrete, in which case we need to run argmax to get the next action.
        TODO: See if you can make this a kind of flag e.g. discrete=True and put logic in train().
        :param QValPreds: The array of the Q value predictions.
        """
        raise NotImplementedError('Please implement a way to return the action from the Q value predictions.')

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
        # TODO: Double check the save behavior to ensure consistent format.
        self.targetModel.save(os.path.join(self.outputDir, self.expName, 'model.tgz'))

    def saveDataArtifacts(self, **kwargs):
        """
        Complementary to the loadDataArtifacts method. Anything that needs extra saving is here. Things
        like plots, csvs, etc.
        :param kwargs:
        :return:
        """
        raise NotImplementedError('Please implement saving custom artifacts. Allowed to put pass.')

    def playGame(self):
        """
        How we play the game. This is different than the QTable function because now, our prediction
        has to now return states that can be passed into the resident model.
        :return:
        """

    def train(self, batchSize, trainSteps, resetEnv=False):
        """
        The main method which will train the network, updating the Q values. This can be similar to the
        playGame function. Thus, for DQNs, it isn't advisable to use playGame, unless you are actually
        planning a play through using the network predictions. The loop here is the same as the actual
        play through.

        :param batchSize: The batch size of the model. The model samples this many
        steps from the buffer to use for training the model.
        :param trainSteps: The number of batches to train for.
        :param resetEnv: Whether to reset the environment, meaning to start over the
        ongoing game...
        :return: None
        """
        if resetEnv:
            _ = self.env.reset()
            self.turnIndex = 0
        self.agentToPlay = self.agentPlayOrder[self.turnIndex]
        # Unlike with playGame where we play until game over,
        # here we just loop for the number of train steps.
        # As with playGame, we use the encodeCurrentState method from
        # the environment.
        for _ in range(trainSteps):
            currentEncodedState = self.env.encodeCurrentState()
            # Calculate the next action using epsilon greedy.
            # Either choose a random action or pass the current state
            # through the model...
            if np.random.rand() < self.epsilon:
                action = self.env.agents[self.agentToPlay].chooseRandomAction()
            else:
                # Pass through the model, make sure to add a batch dimension...
                prediction = self.targetModel(np.expand_dims(currentEncodedState, axis=0))
                # These are the Q values. Take the maximum one...
                action = self.chooseActionFromPrediction(prediction)
            # Step forward in the environment using the action.
            nextRawState, reward, gameOver = self.env.stepForward(self.agentToPlay, action)
            # Call encodeCurrentState(), now this is wrt to the nextState...
            nextEncodedState = self.env.encodeCurrentState()
            # Append this to the experience replay.
            self.replay.add((currentEncodedState, self.agentToPlay, reward, nextEncodedState, gameOver))
            # Go to the next player/agent...
            self.turnIndex = (self.turnIndex + 1) % len(self.agentPlayOrder)
            self.agentToPlay = self.agentPlayOrder[self.turnIndex]
            # If we got a game over, then we need to reset the environment...
            # ...and reset the player...
            if gameOver:
                _ = self.env.reset()
                self.turnIndex = 0
                self.agentToPlay = self.agentPlayOrder[self.turnIndex]
            ####################
            ###### TRAIN #######
            # We sample a batch from the experience replay, but only
            # if the buffer is full...Otherwise, skip ahead to
            # the next turn...
            if len(self.replay.experience) != self.replay.expSize:
                continue
            states, actions, rewards, nextStates, dones = self.replay.sample(batchSize)
            # Before we can actually train a batch, we need to calculate
            # the current Q values, and match them to the target
            # Q values we want, based on the rewards and the max Q values
            # of the next states.
            currentQVals = self.targetModel(states)
            nextQVals = self.targetModel(nextStates)
            # In order to train properly, the target "y values" need to have the same
            # shape as the model's output. Since we should only change the weights
            # for the selected action, the rest should be EXACTLY the same as input,
            # so their loss would evaluate to 0.
            # Grab the indices where the next Q value is maximum...
            nextQValMaxLocs = np.argmax(nextQVals, axis=1)
            # Multiply these by gamma and (1 - dones), so the game over flags automatically
            # zero out the done states, leaving the rewards...
            expectedQVals = rewards + (1 - dones) * self.gamma * nextQValMaxLocs
            # Now assign these values to the cells corresponding to the action taken...
            currentQVals[np.arange(currentQVals.shape[0]), actions] = expectedQVals
            # Now train on batch. Because only the action-taken cells changed, only
            # those will have non-zero loss and only those weights will change.
            self.targetModel.train_on_batch(x=states, y=currentQVals)
