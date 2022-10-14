'''
This file defines the environment. It takes the current states and then returns the next state.
'''
import numpy as np
import math
import copy

from config import Configurations

class Environment:
    def __init__(self):
        '''
        These values are done for calculations. Not optimum but the rocket will reach the end.
        '''
        #### Rocket Params
        self.R_E = 6378.4 # in km
        self.DIST_EARTH_END = 145 # in km
        self.R_SE = self.R_E + self.DIST_EARTH_END # in km # Distance of 
        self.L = 1005  # in km
        self.R_SO = self.R_SE + self.L # radius of skyhook's COM
        self.H = 6.7 # in km
        self.g0 = 0.00981 # in km/s^-2
        self.T_max = 3843.425
        self.initial_mass = 408900
        self.fuel_mass = 277300
        self.burnout_mass = 131600
        self.u_e = 3
        self.rho0 = 1.752e+9 # in kg/km^3

        #### Drone Params
        self.D_amax =  0.01 # Maximum accln provided by drone's thruster slighlty greater than g0
        
        #### Skyhook Params
        self.OMEGA = 0.00096657
        self.omega = 0.006765

    def runge_kutta_step(self, f, x, *args):
        k = self.dt
        k1  = k*f(x, *args)
        k2 = k*f(x + 0.5*k1, *args);
        k3 = k*f(x + 0.5*k2, *args);
        k4 = k*f(x + k3, *args);
        return x + (1/6.0) * (k1 + 2*k2+ 2*k3 + k4)
    
    def skyhook_dynamics(self, x):
        '''
        x --> [x, theta, phi]
        '''
        return np.array([0, self.OMEGA, self.omega])
    
    def rocket_dynamics(self, x, *args):
        '''
        x --> [v, r, m]
        '''
        v = x[0]
        r = x[1]
        m = x[2]
        u = args[0]
        T = self.T_max * u
        g = self.g0 * self.R_E ** 2 / r**2
        D = (v**2)*1.732*.1*np.pi*0.05**2*np.exp(-(r - self.R_E)/self.H)/4
        v_dot = (T - D)/m - g
        r_dot = v
        m_dot = -T/self.u_e
        return np.array([v_dot, r_dot, m_dot])

    def drone_dynamics(self, X, *args):
        '''
        x ----> [x, y, x_dot, y_dot]
        '''
        x = X[0]
        y = X[1]
        x_dot = X[2]
        y_dot = X[3]
        F_x = self.D_amax * args[0]
        F_y = self.D_amax * args[1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(y/x)
        g = self.g0*(self.R_E**2)/(r**2)
        x_ddot = F_x - g * np.cos(theta)
        y_ddot = F_y - g * np.sin(theta)

        return np.array([x_dot, y_dot, x_ddot, y_ddot])
    
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

    def step(self, states, actions):
        next_states = copy.deepcopy(states)
        self.dt = Configurations.TIME_INTERVAL
        if states["skyhook"][1] > Configurations.TIME_INTERVAL_CHANGE_PARAM:
            self.dt = Configurations.TIME_INTERVAL_NEW
        next_states["time"] = states["time"] + self.dt

        next_states["skyhook"] = self.runge_kutta_step(self.skyhook_dynamics, states["skyhook"])
        if states["drone"]["mu"] == 1:
            if np.mean(actions["drone"]) < 1e-3:
                next_states["drone"]["state"] = self.get_skyhook_end_position(next_states["skyhook"])
                next_states["drone"]["mu"] = 1
            else:
                next_states["drone"]["state"] = self.runge_kutta_step(self.drone_dynamics, states["drone"]["state"], actions["drone"][0], actions["drone"][1])
                next_states["drone"]["mu"] = 0
        else:
            next_states["drone"]["state"] = self.runge_kutta_step(self.drone_dynamics, states["drone"]["state"], actions["drone"][0], actions["drone"][1])
        
        if states["rocket"]["check_flag"][0] ==0 and states["rocket"]["check_flag"][1] == 0 and actions["rocket"] == 1:
            next_states["rocket"]["check_flag"][0] = 1
        
        if states["rocket"]["check_flag"][0] == 1 and actions["rocket"] == 0:
            states["rocket"]["check_flag"][1] = 1

        if next_states["rocket"]["check_flag"][0] == 0:
            # case when rocket hasn't been ignited yet.
            next_states["rocket"]["state"] = states["rocket"]["state"]
        else:
            next_states["rocket"]["state"] = self.runge_kutta_step(self.rocket_dynamics, states["rocket"]["state"], actions["rocket"])

        drone_reward, drone_done = self.get_drone_reward(states, next_states, actions)
        rocket_reward, rocket_done = self.get_rocket_reward(states, next_states, actions)

        skyhook_done = next_states["skyhook"][1] > 1.5725

        return next_states, drone_reward, rocket_reward, drone_done, rocket_done, skyhook_done
    
    def check_drone_constraint(self, next_states):
        skyhook_end_position = self.get_skyhook_end_position(next_states["skyhook"])
        drone_position = next_states["drone"]["state"]
        distance = np.sqrt((skyhook_end_position[0] - drone_position[0])**2 + (skyhook_end_position[1] - drone_position[1])**2) 
        if distance > 1:
            return True
        return False


    def get_drone_reward(self, states, next_states, actions):
        # check if the drone within skyhook or not.
        if next_states["drone"]["mu"] == 1:
            return 1 - np.mean(np.abs(actions["drone"])), True
    

        # check if nan values encountered or not.
        if np.any(np.isnan(next_states["drone"]["state"])):
            return -5 - np.mean(np.abs(actions["drone"])), True
        
        # in case of normal cases.
        drone_reward = 0
        e_p = np.sqrt((states["drone"]["state"][0])**2 + (states["drone"]["state"][1] - states["rocket"]["state"][1])**2)/Configurations.DRONE_DIST_REF
        e_n = np.sqrt((next_states["drone"]["state"][0])**2 + (next_states["drone"]["state"][1] - next_states["rocket"]["state"][1])**2)/Configurations.DRONE_DIST_REF
        e_v = np.sqrt((next_states["drone"]["state"][2])**2 + (next_states["drone"]["state"][3] - next_states["rocket"]["state"][0])**2)/Configurations.DRONE_VELOCITY_REF
        
        velocity_component = e_v / (e_n + 1) 
        if np.sign(e_p - e_n) >= 0:
            drone_reward += 1
        else:
            drone_reward -= 2
        
        drone_reward += (1 - e_n)
        drone_reward -= velocity_component

        drone_reward += (1 - np.mean(np.abs(actions["drone"])))

        # check for length constraint of the drone < 1km from skyhook end position
        if self.check_drone_constraint(next_states):
            drone_reward -= 5
            return drone_reward, True
        
        return drone_reward, False
    
    def get_rocket_reward(self, states, next_states, actions):
        # check if nan values encountered or not.
        if np.any(np.isnan(next_states["rocket"]["state"])):
            return -5, True
        
        e_p = np.sqrt((self.R_SE * np.cos(states["skyhook"][1]))**2 + (self.R_SE * np.sin(states["skyhook"][1])- states["rocket"]["state"][1])**2)/Configurations.ROCKET_DIST_REF
        e_n = np.sqrt((self.R_SE * np.cos(next_states["skyhook"][1]))**2 + (self.R_SE * np.sin(next_states["skyhook"][1])- next_states["rocket"]["state"][1])**2)/Configurations.ROCKET_DIST_REF
        e_v = np.abs(next_states["rocket"]["state"][0])/Configurations.ROCKET_VELOCITY_REF

        velocity_component = e_v / (e_n + 1) 

        rocket_reward = 0
        if np.sign(e_p - e_n) >= 0:
            rocket_reward += 1
        else:
            rocket_reward -= 2
        
        rocket_reward += (1 - e_n)
        rocket_reward -= velocity_component

        # Can't restart the rocket
        if next_states["rocket"]["check_flag"][0] == 1 and next_states["rocket"]["check_flag"][1] == 1 and actions["rocket"] == 1:
            return rocket_reward - 2, True
        
        # Rocket crashed
        if next_states["rocket"]["state"][1] < 0:
            return rocket_reward - 4, True
        
        # Reached appropriate position
        if e_n * Configurations.ROCKET_DIST_REF < 0.001:
            if next_states["rocket"]["state"][0] < 0.001:
                print("Mission Successful")
                return rocket_reward + 10, True
            else:
                return rocket_reward - 2, True
        
        # Mass went negative.
        if next_states["rocket"]["state"][2] < 0:
            return rocket_reward - 4, True
        
        return rocket_reward, False




        

        



        

                 


