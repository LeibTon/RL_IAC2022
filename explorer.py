import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import json
from config import Configurations


###############################
# REF: https://bit.ly/3RkXEtg
###############################

class Explorer:
    def __init__(self, agent_type = "train"):
        self.agent_type = agent_type
        self.R_E = 6378.4 # in km
        self.DIST_EARTH_END = 145
        self.R_SE = self.R_E + self.DIST_EARTH_END # in km # Distance of 
        self.L = 1005  # in km
        self.R_SO = self.R_SE + self.L # radius of skyhook's COM
        self.initial_mass = 408900
        # time settings for initialising this the initial conditions.
        self.OT = 228 # second when the rocket should be fired. Calculated ideally using the formulae in rocket propulsion class.
        self.LT = self.OT - 25
        self.UT = self.OT + 75
        self.OMEGA = 0.00096657
        self.omega = 0.006765
        self.total_drone_reward = 0
        self.total_rocket_reward = 0
        self.state_storage_for_priority = []
    
    def get_skyhook_end_position(self, skyhook_state):
        r = skyhook_state[0]
        theta = skyhook_state[1]
        phi = skyhook_state[2]
        r1x = r*np.cos(theta) - self.L*np.cos(phi)
        r1y = r*np.sin(theta) - self.L*np.sin(phi)
        theta_1 = np.arctan2(r1y, r1x)
        v_x = - self.OMEGA * r * np.sin(theta) + self.omega *self.L * np.sin(phi)
        v_y =   self.OMEGA * r * np.cos(theta) - self.omega *self.L * np.cos(phi)
        return np.array([r1x, r1y, v_x, v_y])

    def init(self):
        # Complete this function to intialise the intial values of skyhook, rocket and everything
        if self.agent_type != "train":
            time = 228
        else:
            time = np.random.rand()*(self.UT - self.LT) + self.LT

        skyhook_state = np.array([self.R_SO, np.pi/2 - time*self.OMEGA, np.pi/2 - time*self.omega])

        self.state = {
            "skyhook": skyhook_state,
            "rocket": {
                "state": np.array([self.R_E, 0.001, self.initial_mass]),
                "check_flag": np.array([0, 0])
            },
            "drone": {
                "state": self.get_skyhook_end_position(skyhook_state),
                "mu": 1
            },
            "time": 0
        }

    def get_model_states(self, states, cat = False):
        '''
        This function transfers the dictionary to something valuable.
        '''
        drone_state = np.array([states["drone"]["state"][0], states["drone"]["state"][1] - states["rocket"]["state"][1], states["drone"]["state"][2], states["drone"]["state"][3] - states["rocket"]["state"][0]])
        rocket_state = np.array([self.R_SE * np.cos(states["skyhook"][1]), self.R_SE * np.sin(states["skyhook"][1])- states["rocket"]["state"][1], states["rocket"][0], states["rocket"][2]])
        if cat:
            return np.concatenate([drone_state, rocket_state])
        else:
            return rocket_state, drone_state
    
    def get_action(self, model):
        rocket_state, drone_state = self.get_model_states(self.state, False)
        rocket_action = model.RI_controller.actor(rocket_state).detach().cpu().to_numpy()
        drone_action = model.DI_controller.actor(drone_state).detach().cpu().to_numpy()

        return {"rocket": rocket_action, "drone": drone_action}
    
    def get_next_state(self, environment, action, replay_buffer):
        """"
        This will get data from envionrment and then proceed to store the after effects somewhere.
        """
        action_modified = deepcopy(action)
        action_modified["rocket"] = np.argmax(action["rocket"])
        next_states, drone_reward, rocket_reward, drone_done, rocket_done, skyhook_done = environment.step(self.states, action_modified)
        exp_state = self.get_model_states(self.states, cat = True)
        exp_next_state = self.get_model_states(next_states, cat = True)
        exp_reward = np.array([rocket_reward, drone_reward])
        exp_done = np.array([rocket_done, drone_done])
        exp_action = np.zeros(4)
        exp_action[0:2] = action["rocket"]
        exp_action[2:4] = action["drone"]
        self.total_drone_reward += drone_reward
        self.total_rocket_reward += rocket_reward
        self.state_storage_for_priority.append([exp_state, exp_action, exp_reward, exp_next_state, exp_done])

        # Priority buffer addition here.
        if skyhook_done or drone_done or rocket_done:
            if max(self.total_drone_reward, self.total_rocket_reward) >= Configurations.MAX_TOTAL_REWARD:
                Configurations.MAX_TOTAL_REWARD = max(self.total_drone_reward, self.total_rocket_reward)
                for experience in self.state_storage_for_priority:
                    replay_buffer.add_priority_buffer(experience)
                self.total_drone_reward = 0
                self.total_rocket_reward = 0
                self.state_storage_for_priority = []

        replay_buffer.add_buffer([exp_state, exp_action, exp_reward, exp_next_state, exp_done])
        self.state = next_states
        return drone_reward, rocket_reward, skyhook_done or drone_done or rocket_done



class TestAgent(Explorer):
    def __init__(self):
        super(TestAgent).__init__("test")
    
    def run(self, model, environment, replay_buffer):
        self.init()
        self.store_data = []
        self.store_rewards = []
        self.action_data = []
        done = False
        self.store_data.append(self.state)
        while not done:
            action = self.get_action(model)
            drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
            self.store_rewards.append([rocket_reward, drone_reward])
            self.store_data.append(self.state)
            self.action_data.append(action)
        self.print_info()
        if Configurations.CURRENT_EPISODE in [1000, 15000, 30000, 50000, 60000, 70000, 90000, 100000]:
            self.save_data()

    def print_info(self):
        '''
        Display whatever you want to display to the person.
        '''
        episode_number = Configurations.CURRENT_EPISODE
        rocket_total_reward = sum([i[0] for i in self.store_rewards])
        drone_total_reward = sum([i[0] for i in self.store_rewards])
        rocket_final_velocity = self.store_data[-1]["rocket"]["state"][0]
        rocket_final_height = self.store_data[-1]["rocket"]["state"][1]
        drone_rocket_position_error = np.sqrt((self.store_data[-1]["drone"]["state"][0])**2 + (self.store_data[-1]["drone"]["state"][1] - self.store_data[-1]["rocket"]["state"][1])**2)
        drone_rocket_velocity_error = np.sqrt((self.store_data[-1]["drone"]["state"][2])**2 + (self.store_data[-1]["drone"]["state"][3] - self.store_data[-1]["rocket"]["states"][0])**2)
        print("#################################################")
        print("EPISODE NUMBER:", episode_number)
        print("Rocket Reward:", rocket_total_reward)
        print("Drone Reward:", drone_total_reward)
        print("Rocket Final Velocity:", rocket_final_velocity)
        print("Rocket Final Height:", rocket_final_height)
        print("Distance between rocket and drone:", drone_rocket_position_error)
        print("Rocket Drone velocity error:", drone_rocket_velocity_error)
        print("#######################################")
        with open(Configurations.DATA_SAVE_FILE_NAME, "a") as save_file:
            save_file.write(str(episode_number) +" " + str(rocket_total_reward) + " " +  str(drone_total_reward) + " " + str(rocket_final_velocity) + " " + str(rocket_final_height) + " " + str(drone_rocket_position_error)  + " " + str(drone_rocket_velocity_error) +'\n')
    
    
    def save_data(self):
        '''
        Saves the data for research paper purpose.
        '''
        with open("data/action_data_episode_"+Configurations.CURRENT_EPISODE, 'w') as fout:
            json.dump(self.action_data)
        
        with open("data/state_data_episode_"+Configurations.CURRENT_EPISODE, 'w') as fout:
            json.dump(self.store_data)

    
class BetaExplorer(Explorer):
    def __init__(self, factor):
        super(BetaExplorer).__init__("train")
        self.factor = factor
    
    def run(self, model, environment, replay_buffer):
        noise_scale = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        action = self.get_action(model)
        rocket_action = action["rocket"]
        sign = np.sign(rocket_action)
        alpha = 1/noise_scale
        value = 0.5 + rocket_action / 2
        beta = alpha * (1 - value)/value
        beta = beta + 1*((alpha - beta)/alpha)
        sample = np.random.beta(alpha, beta)
        sample = sign * sample + (1 - sign)/2
        action["rocket"] = sample

        drone_action = action["drone"]
        sign = np.sign(drone_action)
        alpha = 1/noise_scale
        value = 0.5 + drone_action / 2
        beta = alpha * (1 - value)/value
        beta = beta + 1*((alpha - beta)/alpha)
        sample = np.random.beta(alpha, beta)
        sample = sign * sample + (1 - sign)/2

        action["drone"] = 2 * sample - 1

        drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
        
        if done:
            self.init()
        return drone_reward, rocket_reward


class OUExplorer(Explorer):
    def __init__(self, factor):
        super(OUExplorer).__init__("train")
        self.action_size = 2
        self.factor = factor
        self.mu_drone = 0
        self.mu_rocket = 0.5
        self.theta = Configurations.TIME_INTERVAL
        self.sigma = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        self.state_drone = np.ones(self.action_size)*self.mu_drone
        self.state_rocket = np.ones(self.action_size)*self.mu_rocket
        self.reset()
    
    def reset(self):
        self.sigma = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        self.state_drone = np.ones(self.action_size)*self.mu_drone
        self.state_rocket = np.ones(self.action_size)*self.mu_rocket
        
    def run(self, model, environment, replay_buffer):
        self.state_rocket = self.state_rocket  + self.theta * (self.mu_rocket - self.state_rocket) + self.sigma * np.random.randn(2)
        self.state_drone = self.state_drone + self.theta * (self.mu_drone - self.state_drone) + self.sigma * np.random.randn(2)        
        action = self.get_action(model)
        action["rocket"] += self.state_rocket
        action["rocket"] = np.clip(action["rocket"], 0, 1)
        action["drone"] += self.state_drone
        action["drone"] = np.clip(action["drone"], -1, 1)
        drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
        if done:
            self.init()
            self.reset()
        return drone_reward, rocket_reward


class GaussianExplorer(Explorer):
    def __init__(self, factor):
        super(GaussianExplorer).__init__("train")
        self.factor = factor
    
    def run(self, model, environment, replay_buffer):
        action = self.get_action(model)
        sigma = 2/3 * (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        action["rocket"] += np.random.normal(0.5, sigma/2, 2)  # here half sigma is considered because the value of rocket action rnages between 0-1
        action["rocket"] = np.clip(action["rocket"], 0, 1)
        action["drone"] += np.random.normal(0, sigma, 2)
        action["drone"] = np.clip(action["drone"], -1, 1)
        drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
        if done:
            self.init()
        return drone_reward, rocket_reward
    



class epsilonExplorer(Explorer):
    def __init__(self, factor):
        super(epsilonExplorer).__init__("train")
        self.factor = factor
    
    def run(self, model, environment, replay_buffer):
        action = self.get_action(model)
        epsilon = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        if np.random.rand() < epsilon:
            rocket_action = np.random.choice([0,1])
            if rocket_action == 0:
                action["rocket"] = np.array([1, 0])
            else:
                action["rocket"] = np.array([0, 1])
            action["drone"] = np.array([np.random.rand()*2 - 1, np.random.rand()*2 - 1])
        drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
        if done:
            self.init()
        return drone_reward, rocket_reward
        