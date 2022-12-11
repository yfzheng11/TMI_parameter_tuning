# Constants defining our neural network
action_repr = {
    '0': 1.5,
    '1': 1.1,
    '2': 1.0,
    '3': 0.9,
    '4': 0.5,
}

params = {
    'num_iters_for_recon': 10,
    'NPixel': 128,
    'patch_obs': 9,
    'patch_rew': 5,
    'batch_size': 128,
    'ac_dim': 5,
    'num_epoches': 300,
    'scalar_log_freq': -1,
    'learning_starts': 1,
    'learning_freq': 1,
    'target_update_freq': 15,
    'action_repr': action_repr,
    'recon_param_lb': 1e-5,
    'recon_param_ub': 10,
    'num_agent_train_steps_per_iter': 30,
    'replay_buffer_size': 5000000,
    'dqn_discount_rate_gamma': 0.99,
    'dqn_learning_rate': 0.00001,
    'dqn_double_q': True,
    'dqn_network_hidden_size': 128,
    'dqn_grad_norm_clipping': 10,
    'logdir': None,
    'random_seed': 5,
    'save_params': False
}

# NPROJ = 60
# NP = 128
# TRAIN_IMG_NUM = 6
# TEST_IMG_NUM = 6
# PATCH_NUM = NPixel * NPixel
# Train_NUM_total = PATCH_NUM * TRAIN_IMG_NUM
# DISCOUNT_RATE = 0.99
