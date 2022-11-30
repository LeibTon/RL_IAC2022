'''
This file consists the actor and critic classes.
'''
import torch.nn as nn
import torch as th
import torch.nn.functional as F

from config import Configurations


# for initalising all layers of a network at once
def initialize_weights(net, low=-3e-2, high=3e-2):
    for param in net.parameters():
        param.data.uniform_(low, high)

class Actor(nn.Module):
    def __init__(self, action_size, state_size, action_type):
        super(Actor, self).__init__()
        self.action_type = action_type
        self.FC1 = nn.Linear(state_size, Configurations.ACTOR_LAYERS[0])
        self.FC2 = nn.Linear(Configurations.ACTOR_LAYERS[0], Configurations.ACTOR_LAYERS[1])
        self.FC3 = nn.Linear(Configurations.ACTOR_LAYERS[1], action_size)
        initialize_weights(self)
    
    def forward(self, state):
        layer_1 = F.relu(self.FC1(state))
        layer_2 = F.relu(self.FC2(layer_1))
        if self.action_type == "continuous":
            action = th.tanh(self.FC3(layer_2))
        else:
            action = F.softmax(self.FC3(layer_2), dim = -1)
        return action

class Critic(nn.Module):
    def __init__(self, actions_size, states_size, num_atoms):
        super(Critic, self).__init__()
        self.FC1 = nn.Linear(states_size, Configurations.CRITIC_LAYERS[0])
        self.FC2 = nn.Linear(Configurations.CRITIC_LAYERS[0] + actions_size, Configurations.CRITIC_LAYERS[1])
        self.FC3 = nn.Linear(Configurations.CRITIC_LAYERS[1], 1)
        initialize_weights(self)
    
    def forward(self, states, actions, log = False):
        layer_1 = F.relu(self.FC1(states))
        layer_1_cat = th.cat([layer_1, actions], dim = 1)
        layer_2 = F.relu(self.FC2(layer_1_cat))
        return self.FC3(layer_2)
        # Q_probs = self.FC3(layer_2)
        # if log:
        #     return F.log_softmax(Q_probs, dim = -1)
        # else:
        #     return F.softmax(Q_probs, dim = -1) #  softmax converts the Q_probs to valid probabilities (i.e. 0 to 1 and all sum to 1)