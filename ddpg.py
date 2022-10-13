'''
This class builds the learner which consitutes the critic, actor of drone and Q-function of the rocket.
'''

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


from models import Actor, Critic
from config import Configurations


device = th.device("cuda" if th.cuda.is_available() else "cpu")

def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    


class DDPGAgent:
    def __init__(self, action_size, state_size, actions_size, states_size, num_atoms, action_type):
        self.actor = Actor(action_size, state_size, action_type).to(device)
        self.target_actor = Actor(action_size, state_size, action_type).to(device)
        self.critic = Critic(actions_size, states_size, num_atoms).to(device)
        self.target_critic = Critic(actions_size, states_size, num_atoms).to(device)

        self.action_type = action_type
        

        self.actor_optimizer = Adam(self.actor.parameters(), lr = Configurations.ACTOR_LR)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = Configurations.CRITIC_LR)
    
    def hardupdate(self):
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

