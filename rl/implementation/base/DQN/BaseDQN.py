"""
File: BaseDQN.py
Creation Date: 2021-04-01
Location: implementation/base/DQN

A barebones DQN class. Has everything to train a bare model. This is similar to the Qtable
in that this also has a play game function. The reason they are separate, is because of
the way this plays the game is different, since its prediction method for the next action
is neural network based instead of table based.
"""