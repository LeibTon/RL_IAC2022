import numpy as np
import torch as th
from copy import deepcopy
import matplotlib.pyplot as plt
import json
from collections import deque
from config import Configurations

device = th.device("cuda" if th.cuda.is_available() else "cpu")
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
        self.T_max = 3870
        self.fuel_mass = 301000
        self.burnout_mass = 80000
        self.initial_mass = self.fuel_mass + self.burnout_mass
        # time settings for initialising this the initial conditions.
        self.OT = 228 # second when the rocket should be fired. Calculated ideally using the formulae in rocket propulsion class.
        self.LT = self.OT - 25
        self.UT = self.OT + 75
        self.OMEGA = 0.00096657
        self.omega = 0.006765
        self.total_drone_reward = 0
        self.total_rocket_reward = 0
        self.state_storage_for_priority = []
        self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)
        self.init()
    
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
        self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)
        if self.agent_type != "train":
            time = 228
        else:
            time = np.random.rand()*(self.UT - self.LT) + self.LT

        skyhook_state = np.array([self.R_SO, np.pi/2 - time*self.OMEGA, np.pi/2 - time*self.omega])

        self.state = {
            "skyhook": skyhook_state,
            "rocket": {
                "state": np.array([0.001, self.R_E, self.initial_mass]),
                "check_flag": np.array([0, 0])
            },
            "drone": {
                "state": self.get_skyhook_end_position(skyhook_state),
                "mu": 1
            },
            "time": 0
        }
        self.total_drone_reward = 0
        self.total_rocket_reward = 0
        self.state_storage_for_priority = []
        self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)

    def get_model_states(self, states, cat = False):
        '''
        This function transfers the dictionary to something valuable.
        '''
        drone_state = np.array([states["drone"]["state"][0], states["drone"]["state"][1] - states["rocket"]["state"][1], states["drone"]["state"][2], states["drone"]["state"][3] - states["rocket"]["state"][0]])
        rocket_state = np.array([self.R_SE * np.cos(states["skyhook"][1]), self.R_SE * np.sin(states["skyhook"][1])- states["rocket"]["state"][1], states["rocket"]["state"][0], states["rocket"]["state"][2]])
        if cat:
            return np.concatenate([drone_state, rocket_state])
        else:
            return rocket_state, drone_state
    
    def get_action(self, model):
        rocket_state, drone_state = self.get_model_states(self.state, False)
        rocket_state = th.from_numpy(rocket_state).float().to(device)
        drone_state = th.from_numpy(drone_state).float().to(device)
        rocket_action = model.RI_controller.actor(rocket_state).detach().cpu().numpy()
        drone_action = model.DI_controller.actor(drone_state).detach().cpu().numpy()

        return {"rocket": rocket_action, "drone": drone_action}

    def add(self, experience, function):
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
            function(transition)

    
    def get_next_state(self, environment, action, replay_buffer):
        """"
        This will get data from envionrment and then proceed to store the after effects somewhere.
        """
        action_modified = deepcopy(action)
        action_modified["rocket"] = np.argmax(action["rocket"])
        next_state, drone_reward, rocket_reward, drone_done, rocket_done, skyhook_done = environment.step(self.state, action_modified)
        exp_state = self.get_model_states(self.state, cat = True)
        exp_next_state = self.get_model_states(next_state, cat = True)
        exp_reward = np.array([rocket_reward, drone_reward])
        exp_done = np.array([rocket_done, drone_done])
        exp_action = np.zeros(4)
        exp_action[0:2] = action["rocket"]
        exp_action[2:4] = action["drone"]
        self.total_drone_reward += drone_reward
        self.total_rocket_reward += rocket_reward
        self.state_storage_for_priority.append([exp_state, exp_action, exp_reward, exp_next_state, exp_done])

        # Priority buffer addition here.
        self.add([exp_state, exp_action, exp_reward, exp_next_state, exp_done], replay_buffer.add_buffer)
        if skyhook_done or drone_done or rocket_done:
            self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)
            if max(self.total_drone_reward, self.total_rocket_reward) >= Configurations.MAX_TOTAL_REWARD:
                Configurations.MAX_TOTAL_REWARD = max(self.total_drone_reward, self.total_rocket_reward)
                for experience in self.state_storage_for_priority:
                    self.add(experience, replay_buffer.add_priority_buffer)
            self.state_storage_for_priority = []
            self.n_step_buffer = deque(maxlen = Configurations.N_STEPS)

        self.state = next_state
        return drone_reward, rocket_reward, skyhook_done or drone_done or rocket_done



class TestAgent(Explorer):
    def __init__(self):
        super(TestAgent, self).__init__("test")
    
    def run(self, model, environment, replay_buffer):
        self.init()
        self.store_data = []
        self.store_rewards = []
        self.action_data = []
        done = False
        self.store_data.append(self.state)
        self.steps = 0
        while not done:
            action = self.get_action(model)
            drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
            self.store_rewards.append([rocket_reward, drone_reward])
            action["drone"] = action["drone"].tolist()
            action["rocket"] = action["rocket"].tolist()
            state = deepcopy(self.state)
            state["skyhook"] = self.state["skyhook"].tolist()
            state["drone"]["state"] = self.state["drone"]["state"].tolist()
            state["rocket"]["state"] = self.state["rocket"]["state"].tolist()
            state["rocket"]["check_flag"] = self.state["rocket"]["check_flag"].tolist()
            state["drone"]["mu"] = str(self.state["drone"]["mu"])
            state["time"] = str(self.state["time"])
            self.store_data.append(state)
            self.action_data.append(action)
            self.steps += 1
        self.print_info(replay_buffer)
        if Configurations.CURRENT_EPISODE in [1000, 15000, 30000, 50000, 60000, 70000, 90000, 100000]:
            self.save_data()

    def print_info(self, buffer):
        '''
        Display whatever you want to display to the person.
        '''
        episode_number = Configurations.CURRENT_EPISODE
        rocket_total_reward = sum([i[0] for i in self.store_rewards])
        drone_total_reward = sum([i[1] for i in self.store_rewards])
        rocket_final_velocity = self.store_data[-1]["rocket"]["state"][0]
        rocket_final_height = self.store_data[-1]["rocket"]["state"][1]
        drone_rocket_position_error = np.sqrt((self.store_data[-1]["drone"]["state"][0])**2 + (self.store_data[-1]["drone"]["state"][1] - self.store_data[-1]["rocket"]["state"][1])**2)
        drone_rocket_velocity_error = np.sqrt((self.store_data[-1]["drone"]["state"][2])**2 + (self.store_data[-1]["drone"]["state"][3] - self.store_data[-1]["rocket"]["state"][0])**2)
        print("#################################################")
        print("EPISODE NUMBER:", episode_number)
        print("Steps:", self.steps)
        print("Buffer Length", len(buffer.pool_1))
        print("Priority Buffer Length", len(buffer.pool_2))
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
        np.save("data/action_data_episode_"+str(Configurations.CURRENT_EPISODE), self.action_data)
        np.save("data/state_data_episode_"+str(Configurations.CURRENT_EPISODE), self.store_data)

    
class BetaExplorer(Explorer):
    def __init__(self, factor):
        super(BetaExplorer, self).__init__("train")
        self.factor = factor
    
    def run(self, model, environment, replay_buffer):
        noise_scale = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        action = self.get_action(model)
        epsilon = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        if np.random.rand() < epsilon:
            rocket_action = np.random.choice([0,1])
            if rocket_action == 0:
                action["rocket"] = np.array([1, 0])
            else:
                action["rocket"] = np.array([0, 1])

        drone_action = action["drone"]
        sign = np.sign(drone_action)
        alpha = 1/noise_scale
        value = 0.5 + drone_action / 2
        beta = alpha * (1 - value)/value
        beta = np.abs(beta + 1*((alpha - beta)/alpha))
        sample = np.random.beta(alpha, beta)
        sample = sign * sample + (1 - sign)/2

        action["drone"] = 2 * sample - 1

        drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
        
        if done:
            self.init()
        return done


class OUExplorer(Explorer):
    def __init__(self, factor):
        super(OUExplorer, self).__init__("train")
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
        epsilon = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        if np.random.rand() < epsilon:
            rocket_action = np.random.choice([0,1])
            if rocket_action == 0:
                action["rocket"] = np.array([1, 0])
            else:
                action["rocket"] = np.array([0, 1])
        action["drone"] += self.state_drone
        action["drone"] = np.clip(action["drone"], -1, 1)
        drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
        if done:
            self.init()
            self.reset()
        return done


class GaussianExplorer(Explorer):
    def __init__(self, factor):
        super(GaussianExplorer, self).__init__("train")
        self.factor = factor
    
    def run(self, model, environment, replay_buffer):
        action = self.get_action(model)
        sigma = 2/3 * (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        epsilon = (0.99994**Configurations.CURRENT_EPISODE)*self.factor
        if np.random.rand() < epsilon:
            rocket_action = np.random.choice([0,1])
            if rocket_action == 0:
                action["rocket"] = np.array([1, 0])
            else:
                action["rocket"] = np.array([0, 1])
        action["drone"] += np.random.normal(0, sigma, 2)
        action["drone"] = np.clip(action["drone"], -1, 1)
        drone_reward, rocket_reward, done = self.get_next_state(environment, action, replay_buffer)
        if done:
            if Configurations.CURRENT_EPISODE % 10 == 0:
                self.print_info(replay_buffer)
            self.init()
        return done

    def print_info(self, buffer):
        '''
        Display whatever you want to display to the person.
        '''
        episode_number = Configurations.CURRENT_EPISODE
        rocket_total_reward = self.total_rocket_reward
        drone_total_reward = self.total_drone_reward
        rocket_final_velocity = self.state["rocket"]["state"][0]
        rocket_final_height = self.state["rocket"]["state"][1]
        drone_rocket_position_error = np.sqrt((self.state["drone"]["state"][0])**2 + (self.state["drone"]["state"][1] - self.state["rocket"]["state"][1])**2)
        drone_rocket_velocity_error = np.sqrt((self.state["drone"]["state"][2])**2 + (self.state["drone"]["state"][3] - self.state["rocket"]["state"][0])**2)
        with open("gaussian_explorer_"+str(self.factor)+"_data.txt", "a") as save_file:
            save_file.write(str(episode_number) +" " + str(rocket_total_reward) + " " +  str(drone_total_reward) + " " + str(rocket_final_velocity) + " " + str(rocket_final_height) + " " + str(drone_rocket_position_error)  + " " + str(drone_rocket_velocity_error) +'\n')


class epsilonExplorer(Explorer):
    def __init__(self, factor):
        super(epsilonExplorer, self).__init__("train")
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
        return done
        