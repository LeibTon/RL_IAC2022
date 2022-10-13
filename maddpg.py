from config import Configurations
from ddpg import DDPGAgent
import torch as th
import numpy as np

from config import Configurations

device = th.device("cuda" if th.cuda.is_available() else "cpu")


########################################################
# Ref: https://bit.ly/3UMCfMr
##########################################################

class MADDPG:
    def __init__(self):
        self.RI_controller = DDPGAgent(Configurations.ROCKET_ACTION_SIZE, Configurations.ROCKET_STATE_SIZE, Configurations.ROCKET_ACTION_SIZE + Configurations.DRONE_ACTION_SIZE, Configurations.ROCKET_STATE_SIZE + Configurations.DRONE_STATE_SIZE, Configurations.NUM_ATOMS, "discrete")
        self.DI_controller = DDPGAgent(Configurations.DRONE_ACTION_SIZE, Configurations.DRONE_STATE_SIZE, Configurations.ROCKET_ACTION_SIZE + Configurations.DRONE_ACTION_SIZE, Configurations.ROCKET_STATE_SIZE + Configurations.DRONE_STATE_SIZE, Configurations.NUM_ATOMS, "continuous")
        self.discount_rate = Configurations.N_STEPS_DISCOUNT_FACTOR
        self.n_steps = Configurations.N_STEPS
        self.num_atoms = Configurations.NUM_ATOMS
        self.vmin = Configurations.VMIN
        self.vmax = Configurations.VMAX

        self.update_target = Configurations.UPDATE_TARGETS

        self.atoms = th.linspace(self.vmin, self.vmax, self.num_atoms).to(device)
        self.atoms = self.atoms.unsqueeze(0)
        self.num_steps = 1
        
    def get_actors(self):
        actors = [self.RI_controller.actor, self.DI_controller.actor]
        return actors
    
    def get_target_actors(self):
        target_actors = [self.RI_controller.target_actor, self.DI_controller.target_actor]
        return target_actors
    
    def update_saved_param(self, file):
        '''
        This function loads the saved values.
        '''
        checkpoint = th.load(file)
        self.DI_controller.actor.load_state_dict(checkpoint['model_drone_agent_state_dict'])
        self.DI_controller.critic.load_state_dict(checkpoint['model_drone_critic_state_dict'])
        self.RI_controller.actor.load_state_dict(checkpoint['model_rocket_agent_state_dict'])
        self.RI_controller.critic.load_state_dict(checkpoint['model_rocket_critic_state_dict'])
        self.DI_controller.actor_optimizer.load_state_dict(checkpoint['optimizer_drone_agent_state_dict'])
        self.DI_controller.critic_optimizer.load_state_dict(checkpoint['optimizer_drone_critic_state_dict'])
        self.RI_controller.actor_optimizer.load_state_dict(checkpoint['optimizer_rocket_agent_state_dict'])
        self.RI_controller.critic_optimizer.load_state_dict(checkpoint['optimizer_rocket_critic_state_dict'])

        self.RI_controller.hardupdate()
        self.DI_controller.hardupdate()

        self.DI_controller.actor.eval()
        self.DI_controller.critic.eval()
        self.RI_controller.actor.eval()
        self.RI_controller.critic.eval()
        self.DI_controller.target_actor.eval()
        self.DI_controller.target_critic.eval()
        self.RI_controller.target_critic.eval()
        self.RI_controller.target_actor.eval()

        print("Updated parameters.")


    def save_param(self):
        '''
        This function saves the values.
        '''
        file_path = "model/models_eps_"+Configurations.CURRENT_EPISODE +".pt"
        th.save({
            'model_drone_agent_state_dict': self.DI_controller.actor.state_dict(),
            'model_drone_critic_state_dict': self.DI_controller.critic.state_dict(),
            'model_rocket_agent_state_dict': self.RI_controller.actor.state_dict(),
            'model_rocket_critic_state_dict': self.RI_controller.critic.state_dict() ,
            'optimizer_drone_agent_state_dict': self.DI_controller.actor_optimizer.state_dict(),
            'optimizer_drone_critic_state_dict': self.DI_controller.critic_optimizer.state_dict(),
            'optimizer_rocket_agent_state_dict': self.RI_controller.actor_optimizer.state_dict(),
            'optimizer_rocket_critic_state_dict': self.RI_controller.critic_optimizer.state_dict()
        }, file_path)
        print("Model successfully saved at episode:", Configurations.CURRENT_EPISODE)
    
    def update(self, replay_buffer):
        '''
        Performs a distributional Actor/Critic Calculation and update.
        '''
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample()

        states_tensor = th.from_numpy(states_batch).to(device)
        actions_tensor = th.from_numpy(actions_batch).to(device)
        rewards_tensor = th.from_numpy(rewards_batch).to(device)
        next_states_tensor = th.from_numpy(next_states_batch).to(device)
        dones_tensor = th.from_numpy(dones_batch).to(device)

        states_tensor_rocket = states_tensor[:, :Configurations.ROCKET_STATE_SIZE]
        states_tensor_drone = states_tensor[:, Configurations.ROCKET_STATE_SIZE:]
        actions_tensor_rocket = actions_tensor[:, :Configurations.ROCKET_ACTION_SIZE]
        actions_tensor_drone = actions_tensor[:,Configurations.ROCKET_ACTION_SIZE:]
        rewards_tensor_rocket = rewards_tensor[:, 0]
        rewards_tensor_drone = rewards_tensor[:,1]
        done_tensor_rocket = dones_tensor[:,0]
        done_tensor_drone = dones_tensor[:,1]
        next_states_tensor_rocket = next_states_tensor[:, :Configurations.ROCKET_STATE_SIZE]
        next_states_tensor_drone = next_states_tensor[:, Configurations.ROCKET_STATE_SIZE:]

        target_actions_rocket = self.RI_controller.target_actor(next_states_tensor_rocket)
        target_actions_drone = self.DI_controller.target_actor(next_states_tensor_drone)
        target_actions = th.cat([target_actions_rocket, target_actions_drone], dim = 1)

        #################################
        # updating Rocket Controller DDPG
        ##################################
        # update critic network
        self.RI_controller.critic_optimizer.zero_grad()
        target_probs = self.RI_controller.target_critic(next_states_tensor, target_actions).detach()
        target_dist = self.to_categorical(rewards_tensor_rocket.unsqueeze(-1), target_probs, done_tensor_rocket.unsqueeze(-1))
        log_probs = self.RI_controller.critic(states_tensor, actions_tensor, log = True)
        critic_loss = -(target_dist * log_probs).sum(-1).mean()

        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.RI_controller.critic.parameters(), 1)
        self.RI_controller.critic_optimizer.step()

        if self.num_steps % Configurations.ACTOR_UPDATE == 0:
            # update actor network
            self.RI_controller.actor_optimizer.zero_grad()
            actor_actions = [self.RI_controller.actor(states_tensor_rocket), self.DI_controller.actor(states_tensor_drone).detach()]
            actor_actions = th.cat(actor_actions, dim = 1)
            critic_probs = self.RI_controller.critic(states_tensor, actor_actions)
            expected_reward = (critic_probs * self.atoms).sum(-1)
            actor_loss = -expected_reward.mean()
            actor_loss.backward(retain_graph = True)
            self.RI_controller.actor_optimizer.step()

        
        #################################
        # updating Drone Controller DDPG
        #################################
        # update critic network
        self.DI_controller.critic_optimizer.zero_grad()
        target_probs = self.DI_controller.target_critic(next_states_tensor, target_actions).detach()
        target_dist = self.to_categorical(rewards_tensor_drone.unsqueeze(-1), target_probs, done_tensor_drone.unsqueeze(-1))
        log_probs = self.DI_controller.critic(states_tensor, actions_tensor, log = True)
        critic_loss = -(target_dist * log_probs).sum(-1).mean()

        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.DI_controller.critic.parameters(), 1)
        self.DI_controller.critic_optimizer.step()

        if self.num_steps % Configurations.ACTOR_UPDATE == 0:
            # update actor network
            self.DI_controller.actor_optimizer.zero_grad()
            actor_actions = [self.RI_controller.actor(states_tensor_rocket).detach(), self.DI_controller.actor(states_tensor_drone)]
            actor_actions = th.cat(actor_actions, dim = 1)
            critic_probs = self.DI_controller.critic(states_tensor, actor_actions)
            expected_reward = (critic_probs * self.atoms).sum(-1)
            actor_loss = -expected_reward.mean()
            actor_loss.backward(retain_graph = True)
            self.DI_controller.actor_optimizer.step()
            self.num_steps = 0

        if Configurations.CURRENT_EPISODE % Configurations.UPDATE_TARGETS == 0:
            self.RI_controller.hardupdate()
            self.DI_controller.hardupdate()
        
        self.num_steps+=1
        

    def to_categorical(self, rewards, probs, dones):
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms
        n_steps = self.n_steps
        discount_rate = self.discount_rate

        delta_z = (vmax - vmin)/(num_atoms - 1)
        projected_atoms = rewards + discount_rate**n_steps * atoms * ( 1- dones)
        projected_atoms.clamp_(vmin, vmax)
        b = (projected_atoms - vmin)/delta_z

        precision = 1
        b = th.round(b  * 10 ** precision)/10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        # initialising projected_probs
        projected_probs = th.tensor(np.zeros(probs.size())).to(device)

        # a bit like one-hot encoding but for the specified atoms
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()