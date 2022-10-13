'''
Generates and manages the large experience replay buffer of both pools
'''

import random
import numpy as np
from collections import deque

from config import Configurations

class ReplayBuffer():

    def __init__(self, max_length):
        self.buffer = deque(maxlen = max_length)
        self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)

    def reset(self):
        self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)

    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience):
        '''n-step mechanism'''
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == Configurations.N_STEPS:
            start_state, start_action, start_reward, start_next_state, start_done = self.n_step_buffer[0]
            n_state, n_action, n_reward, n_next_state, n_done = self.n_step_buffer[-1]

            summed_reward = np.zeros(2)
            for i, n_transition in enumerate(self.n_step_buffer):
                state, action, reward, next_state, done = n_transition
                summed_reward += reward * Configurations.N_STEPS_DISCOUNT_FACTOR**(i + 1)
                if np.any(done):
                    break
            transition = [start_state, start_action, summed_reward, n_next_state, n_done]
            self.buffer.append(transition)


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
            if Configurations.CURRENT_EPISODE < 2000:
                priority_size = min(int(0.5 * Configurations.MINI_BATCH_SIZE), len(self.pool_2))
                batch_size = Configurations.MINI_BATCH_SIZE - priority_size
            elif Configurations.CURRENT_EPISODE < 4000:
                priority_size = min(int(0.2 * Configurations.MINI_BATCH_SIZE), len(self.pool_2))
                batch_size = Configurations.MINI_BATCH_SIZE - priority_size
            else:
                priority_size = 0
                batch_size = Configurations.MINI_BATCH_SIZE
            sampled_batch = self.pool_1.sample(batch_size) + self.pool_2.sample(priority_size)
        else:
            sampled_batch = self.pool_1.sample(Configurations.MINI_BATCH_SIZE)
        sampled_batch = np.asarray(sampled_batch)
        states_batch = np.stack(sampled_batch[:,0])
        actions_batch = np.stack(sampled_batch[:,1])
        rewards_batch = sampled_batch[:,2]
        next_states_batch = np.stack(sampled_batch[:,3])
        dones_batch = np.stack(sampled_batch[:,4])

        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
    
    def reset(self):
        self.pool_1.reset()
        self.pool_2.reset()
