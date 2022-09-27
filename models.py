'''
This files consists of the Actor and Critic Network for the Rocket and the drone
For the drone, the actor network recieved the state and calcuate the action.
The drone critic network received the state and action and calculates the q -distribution.
The rocket critic network gives the Q-values which is used using argmax function to give the action.
'''

import torch.nn as nn
import torch as th
import torch.nn.functional as F

from config import Configurations

class RocketCritic(nn.Module):
    def __init__(self):
        super(RocketCritic).__init__()
        
        self.FC1 = nn.Linear(Configurations.ROCKET_STATE_SIZE, Configurations.ROCKET_CRITIC_LAYERS[0])
        self.FC2 = nn.Linear(Configurations.ROCKET_CRITIC_LAYERS[0], Configurations.ROCKET_CRITIC_LAYERS[1])
        self.FC3 = nn.Linear(Configurations.ROCKET_CRITIC_LAYERS[1], Configurations.ROCKET_ACTION_SIZE)
        self.softmax = nn.Softmax(dim = 1)  # Check if this dimension is right?
    def forward(self, state):
        result = self.FC1(state)
        result = nn.LeakyReLU(result)
        result = self.FC2(result)
        result = nn.LeakyReLU(result)
        result = self.FC3(result)
        result = self.softmax(result)
        return result

# may need to change the size of the hidden layers based on if we need to account for all the states and actions.
class DroneCritic(nn.Module):
    def __init__(self):
        super(DroneCritic).__init__()

        self.FC1 = nn.Linear(Configurations.DRONE_STATE_SIZE + Configurations.ROCKET_STATE_SIZE, Configurations.DRONE_CRITIC_LAYERS[0])
        self.FC2 = nn.Linear(Configurations.DRONE_CRITIC_LAYERS[0] + Configurations.DRONE_ACTION_SIZE + Configurations.ROCKET_ACTION_SIZE, Configurations.DRONE_CRITIC_LAYERS[1])
        self.FC3 = nn.Linear(Configurations.DRONE_CRITIC_LAYERS[1], 1)
    
    def forward(self, state, action):
        x = self.FC1(state)
        x = F.leaky_relu(x)
        combined = th.cat([x, action], 1)
        x = F.leaky_relu(self.FC2(combined))
        return self.FC3(x)

class DroneActor(nn.Module):
    def __init__(self):
        super(DroneActor).__init__()

        self.FC1 = nn.Linear(Configurations.DRONE_STATE_SIZE, Configurations.DRONE_ACTOR_LAYERS[0])
        self.FC2 = nn.Linear(Configurations.DRONE_ACTOR_LAYERS[0], Configurations.DRONE_ACTOR_LAYERS[1])
        self.FC3 = nn.Linear(Configurations.DRONE_ACTOR_LAYERS[1], Configurations.DRONE_ACTION_SIZE)

    def forward(self, state):
        x = F.leaky_relu(self.FC1(state))
        x = F.leaky_relu(self.FC2(x))
        return F.tanh(self.FC3(x))



