'''
Main script that runs the MADDPG model.
-- Load existing model param if exists.
-- Collect data from the environment all of them.
-- Train the model based on the data.
-- Then again do the same.
-- No parallel currently.
-- Simple problem.
'''


from config import Configurations
from environment import Environment
from replay_buffer import ExperiencePool
from maddpg import MADDPG
from explorer import TestAgent, OUExplorer, GaussianExplorer,epsilonExplorer, BetaExplorer
from tqdm import tqdm
import numpy as np

env = Environment()
pool = ExperiencePool()
model = MADDPG()
ou_explorers = [OUExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
gaussian_explorers = [GaussianExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
epsilon_explorers = [epsilonExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
beta_explorers = [BetaExplorer(Configurations.EXPLORER_SETTINGS[i]) for i in range(6)]
test_agent = TestAgent()
agents = ou_explorers + gaussian_explorers + epsilon_explorers + beta_explorers
flags = [False for agent in agents]

'''
Change these parameters:
resume_training: If resuming parameter
file_location: location of saved file
current_episode: based on the file name you can get the episode number.
'''

def get_truth_table():
    check = True
    for flag in flags:
        check = check and flag
    return check

###################################################################
######## These parameters must be changed before starting training.
###################################################################

resume_training = True
file_location = "model/models_eps_185600.pt"
Configurations.CURRENT_EPISODE = 185600
if resume_training:
    model.update_saved_param(file_location)
    print("Resume Training")
else:
    with open(Configurations.DATA_SAVE_FILE_NAME, "w+") as save_file:
        save_file.write('episode_number rocket_reward drone_reward rocket_velocity rocket_position dist_drone_rocket vel_drone_rocket\n')
    print("Restarting Training")


pbar = tqdm(total = Configurations.TOTAL_EPISODES)
pbar.n = Configurations.CURRENT_EPISODE
pbar.refresh()

while Configurations.CURRENT_EPISODE <= Configurations.TOTAL_EPISODES:
    for idx, agent in enumerate(agents):
        result = agent.run(model, env, pool)
        flags[idx] = result or flags[idx]
    loss = model.update(pool)
    if np.isnan(loss):
        print("Critic loss gone into NaN")

    
    if get_truth_table():
        Configurations.CURRENT_EPISODE += 1
        pbar.update(1)
        pbar.refresh()
        flags = [False for agent in agents]

        if Configurations.CURRENT_EPISODE % 10 == 0:
            print("Critic Loss:", loss)
            test_agent.run(model, env, pool)
    
        if Configurations.CURRENT_EPISODE % 100 == 0:
            model.save_param()

        if Configurations.CURRENT_EPISODE % 100 == 0:
            pool.reset_pool1()
pbar.close()




