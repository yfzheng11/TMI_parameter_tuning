# Constants defining our neural network
action_repr = {
    '0': 1.5,
    '1': 1.1,
    '2': 1.0,
    '3': 0.9,
    '4': 0.5,
}

params = {
    'num_iters': 10,
    'NPixel': 128,
    'patch_obs': 9,
    'patch_rew': 5,
    'batch_size': 128,
    'ac_dim': 5,
    'num_epoches': 300,
    'scalar_log_freq': -1,
    'learning_starts': 1,
    'learning_freq': 1,
    'target_update_freq': 1,
    'action_repr': action_repr,

}

# NPROJ = 60
# NP = 128
# TRAIN_IMG_NUM = 6
# TEST_IMG_NUM = 6
# MAXITER_RECON = 30
#
# PATCH_NUM = NPixel * NPixel
# Train_NUM_total = PATCH_NUM * TRAIN_IMG_NUM
# DISCOUNT_RATE = 0.99
# REPLAY_MEMORY = 5000000
# TARGET_UPDATE_FREQUENCY = 15
# MAX_EPISODES = 300
# load_session = 0
# save_session = 1
#
# TRAIN_NUM_ITER = 10
