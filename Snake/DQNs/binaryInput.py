"""
File: binaryInput.py
Location: /Snake/DQNs
Creation Date: 2020-12-30

This file implements a deep Q-learning network
based on the binary input that is shown in the Q-table
approach. Like that one, we will have 11 binary inputs
corresponding to the state, and it will output one of
3 directions, forward, left, or right. To optimize
learning, I'll use the Double Deep Q-learning
approach, with a target network and a prediction network.
"""

import numpy as np
from ..SnakeAgent import SnakeAgent
from ..SnakeEnv import SnakeGame

