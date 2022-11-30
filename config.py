'''
All configurations related to the model are stored here.
'''


class Configurations:
    ######################
    #Environment Settings#
    ######################
    DRONE_STATE_SIZE = 4
    DRONE_ACTION_SIZE = 2
    ROCKET_ACTION_SIZE = 2
    ROCKET_STATE_SIZE = 4
    TIME_INTERVAL = 1
    TIME_INTERVAL_NEW = 0.5
    TIME_INTERVAL_CHANGE_PARAM =  1.568 # value of theta when the TIME_INTERVAL Changes
    DRONE_DIST_REF = 10 # the drone ref is set to 10 km because this is the range when thee drone should start operating
    DRONE_VELOCITY_REF = 5 # the max speed of the drone is expected to be 5km /s
    ROCKET_DIST_REF = 145 # maximum difference between the rocket and ultimate position.
    ROCKET_VELOCITY_REF = 2 # maximum speed of the rocket during the whole operation.


    ##########################
    # Buffer Settings ########
    ##########################
    REPLAY_BUFFER_SIZE = 1000000
    PRIORIY_REPLAY_BUFFER_SIZE = 100000
    N_STEPS = 2
    N_STEPS_DISCOUNT_FACTOR = 0.99
    GAMMA = 0.9
    MINI_BATCH_SIZE = 256
    PRIORITY_BUFFER = True # Turn this on for priority buffer
    MAX_TOTAL_REWARD = 0 # This is initially set to 0 so that only positive reward is added in priority buffer and is very good indicator somehow.
    DATA_SAVE_FILE_NAME = 'learning_data.txt'

    ############################
    #### Agent Settings ########
    ############################
    EXPLORATION_FACTOR = 0.99998
    EXPLORER_SETTINGS = [1, 0.95, 1.05, 0.85, 1.15, 0.75]


    #########################
    # Training Settings #####
    #########################

    CURRENT_EPISODE = 1  # This updates with time.
    TOTAL_EPISODES = 200000
    ACTOR_UPDATE = 20 # Delayed actor update.

    # Hyperparameters
    UPDATE_TARGETS = 200 # After how many episodes target should be updated.
    NUM_ATOMS = 51  # Everywhere there is this value
    ## These values are based on reward somehow.
    VMIN = - 2
    VMAX = 4
    CRITIC_LAYERS = [400, 300] # The model is designed for two hidden layers only
    ACTOR_LAYERS = [400, 300] # The model is designed for two hidden layers only
    ACTOR_LR = 0.001
    CRITIC_LR = 0.001
    # Periodic Events

    SAVE_INTERVALS = 500 # How many episodes to save network weights



    # Buffer Settings

