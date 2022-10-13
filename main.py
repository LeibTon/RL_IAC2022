'''
Main script that runs the MADDPG model.
-- Load existing model param if exists.
-- Collect data from the environment all of them.
-- Train the model based on the data.
-- Then again do the same.
-- No parallel currently.
-- Simple problem.
'''

import numpy as np
import random
import torch as th
import matplotlib.pyplot as plt

from config import Configurations
from environment import Environment
from replay_buffer import ExperiencePool
from maddpg import MADDPG
from explorer import TestAgent, OUExplorer, GaussianExplorer,epsilonExplorer, BetaExplorer

import os
from collections import deque


env = Environment()
pool = ExperiencePool()
model = MADDPG()
ou_explorers = [OUExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
gaussian_explorers = [GaussianExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
epsilon_explorers = [epsilonExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
beta_explorers = [BetaExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
test_agent = TestAgent()
agents = ou_explorers + gaussian_explorers + epsilon_explorers + beta_explorers


'''
Change these parameters:
resume_training: If resuming parameter
file_location: location of saved file
current_episode: based on the file name you can get the episode number.
'''

###################################################################
######## These parameters must be changed before starting training.
###################################################################

resume_training = False
file_location = "something"
Configurations.CURRENT_EPISODE = 1
if resume_training:
    model.update_saved_param(file_location)
    print("Resume Training")
else:
    with open(Configurations.DATA_SAVE_FILE_NAME, "w+") as save_file:
        save_file.write('episode_number rocket_reward drone_reward rocket_velocity rocket_position dist_drone_rocket vel_drone_rocket\n')
    print("Restarting Training")

while Configurations.CURRENT_EPISODE <= Configurations.TOTAL_EPISODES:
    for agent in agents:
        agent.run(model, env, pool)
    model.update()
    if Configurations.CURRENT_EPISODE % 10 == 0:
        test_agent.run()
    
    if Configurations.CURRENT_EPISODE % 100 == 0:
        model.save_param()
    
    Configurations.CURRENT_EPISODE += 1




