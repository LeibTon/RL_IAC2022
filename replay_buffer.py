'''
Generates and manages the large experience replay buffer of both pools
'''

import random
import numpy as np
from collections import deque

from config import Configurations


class ReplayBuffer():

    def __init__(self, max_length):
        self.max_length = max_length
        self.buffer = deque(maxlen = max_length)
        self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)

    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience):
        '''n-step mechanism'''
        self.buffer.append(experience)

    def reset(self):
        self.buffer = deque(maxlen = self.max_length)


    def sample(self, size_needed):
        batch_size = min(size_needed, len(self.buffer))
        sampled_batch = random.sample(self.buffer, batch_size)
        return sampled_batch

class ExperiencePool():
    def __init__(self):
        self.pool_1 = ReplayBuffer(Configurations.REPLAY_BUFFER_SIZE)
        if Configurations.PRIORITY_BUFFER:
            self.pool_2 = ReplayBuffer(Configurations.PRIORIY_REPLAY_BUFFER_SIZE)
    
    def add_buffer(self, experience):
        '''
        experience --> [states, actions, rewards, next_states, dones]
        '''
        self.pool_1.add(experience)
    
    def add_priority_buffer(self, experience):
        self.pool_2.add(experience)

    def sample(self):
        if Configurations.PRIORITY_BUFFER:
            if Configurations.CURRENT_EPISODE < 10000:
                priority_size = min(int(0.5 * Configurations.MINI_BATCH_SIZE), len(self.pool_2))
                batch_size = Configurations.MINI_BATCH_SIZE - priority_size
            elif Configurations.CURRENT_EPISODE < 20000:
                priority_size = min(int(0.2 * Configurations.MINI_BATCH_SIZE), len(self.pool_2))
                batch_size = Configurations.MINI_BATCH_SIZE - priority_size
            else:
                priority_size = 0
                batch_size = Configurations.MINI_BATCH_SIZE
            sampled_batch = self.pool_1.sample(batch_size) + self.pool_2.sample(priority_size)
        else:
            sampled_batch = self.pool_1.sample(Configurations.MINI_BATCH_SIZE)

        states_batch = np.stack([item[0] for item in sampled_batch])
        actions_batch = np.stack([item[1] for item in sampled_batch])
        rewards_batch = np.stack([item[2] for item in sampled_batch])
        next_states_batch = np.stack([item[3] for item in sampled_batch])
        dones_batch = np.stack([item[4] for item in sampled_batch])
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
    def __len__(self):
        return len(self.pool_1)
    
    def reset_pool1(self):
        self.pool_1.reset()
