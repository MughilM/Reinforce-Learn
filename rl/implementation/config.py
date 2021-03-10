"""
File: config.py
Location: /rl/implementation
Creation Date: 2021-02-20

This file contains all appropriate top-level variables, such as directory locations for
outputs, external data files that might be needed, etc. This shouldn't be usually changed, unless
additional external data locations get added and need access...
"""
import os

# Source directory...
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Base directory
BASE_DIR = os.path.dirname(SRC_DIR)

# Experiment output path
EXP_OUTPUT_DIR = os.path.join(BASE_DIR, 'experimentResults/')

# Print out each location for debugging purposes...
print(f'Source Directory\t\t{SRC_DIR}')
print(f'Base Directory\t\t\t{BASE_DIR}')
print(f'Experiment Output Directory\t{EXP_OUTPUT_DIR}')
