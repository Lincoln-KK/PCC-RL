# PCC-RL
This fork examines different action and reward functions for the Reinforcement learning environment for the Performance-oriented Congestion Control
project. It examines different approaches to designing the action space. It compares useing direct setting of the transmission rate is superior to using a parameterized approach. It examines turning hte action space to discrete. The reward function coefficients are also modified and compared.

## Overview
This repo contains the gym environment required for training reinforcement
learning models used in the PCC project along with the Python module required to
run RL models in the PCC UDT codebase found at github.com/PCCProject/PCC-Uspace.


## Training
Proximal Policy Optimization is used from Stablebaselines3 to train the RL. Bash scripts are used to train on different environment parameters.
Wandb is used for logging. To setup the requirements use `pip install -r requirements.txt`

## Testing Models

The models are tested within the training interval in order to examine the effect of each of the changes made.
