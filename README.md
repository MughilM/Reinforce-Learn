# Reinforcement Learning
## Master Branch
The master branch has the most up-to-date stable version of my projects. Each
game should be able to be played based on the appropriate commands
## Reorganization
As I was adding different ways of playing the game and consequently, different
ways of training the Q-learning, I realized I was running each file standalone.
As a result, methods between the files were being extremely similar, and I was
throwing common methods that didn't need specific objects into a `utils`
folder. This branch aims to modularize everything, so that future game
implementations of other games can be implemented in an extremely similar way.
Almost like a plug-and-play. 

When modularization happens, this obviously means an object-oriented
approach.
