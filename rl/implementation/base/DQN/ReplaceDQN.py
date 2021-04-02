"""
File: ReplaceDQN.py
Creation Date: 2021-04-01
Location: implementation/base/DQN

This file implements a type of deep Q network used for reinforcement learning training.
I've called Replace DQN, where there are actually two copies of the same network.
One is actually trained (called the target network), and one is used for predictions
of the Q values of the next state (called the prediction network). Every so often,
the weights of the prediction network get overwritten by the weights of the target network.
This is so training is more stable, since training for Q values constitutes aiming at a moving target.
Of course, the network has to be defined separately.
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class ReplaceDQN:
    def __init__(self):
        pass

    def createModel(self) -> Model:
        """
        The method used to create the model for target and prediction.
        :return: A TensorFlow Model object.
        """
        raise NotImplementedError('Please implement the createModel function so we can train :)')

    def preprocessBatch(self):
        """
        Inevitably, the encoded states probably will to be able to be forwarded through the model as is.
        Therefore, this method is included so that the user can define how to preprocess
        the included game memory so that it fits the model. Both the input and target output are
        produced here.
        :return:
        """
        raise NotImplementedError('Please implement a way to preprocess the batch of memory.')