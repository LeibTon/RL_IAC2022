'''
All configurations related to the model are stored here.
'''

class Configurations:
    ######################
    #Environment Settings#
    ######################
    DRONE_STATE_SIZE = 4
    DRONE_ACTION_SIZE = 2
    ROCKET_ACTION_SIZE = 1
    ROCKET_STATE_SIZE = 3


    ##########################
    # Buffer Settings ########
    ##########################
    REPLAY_BUFFER_SIZE = 1000000
    PRIORIY_REPLAY_BUFFER_SIZE = 100000
    MINI_BATCH_SIZE = 256
    PRIORITY_BUFFER = False # Turn this on for priority buffer



    #########################
    # Training Settings #####
    #########################

    # Hyperparameters
    ROCKET_CRITIC_LAYERS = [400, 300] # The model is designed for two hidden layers only
    DRONE_CRITIC_LAYERS = [400, 300] # The model is designed for two hidden layers only
    DRONE_ACTOR_LAYERS = [400, 300] # The model is designed for two hidden layers only



    # Periodic Events


    # Buffer Settings

