# Constants defining our neural network
action_repr = {
    '0': 1.5,
    '1': 1.1,
    '2': 1.0,
    '3': 0.9,
    '4': 0.5,
}

params = {
    'num_epoches': 300,
    'num_agent_train_steps_per_iter': 20,
    'num_iters_for_recon': 50,  # 50
    'NPixel': 128,
    'patch_obs': 9,
    'patch_rew': 5,
    'batch_size': 128,
    'ac_dim': 5,
    'scalar_log_freq': 1,
    'learning_starts': 1,
    'learning_freq': 1,
    'target_update_freq': 15,
    'action_repr': action_repr,
    'recon_param_lb': 1e-5,
    'recon_param_ub': 10,
    'replay_buffer_size': 5000000,
    'dqn_discount_rate_gamma': 0.99,
    'dqn_learning_rate': 0.001,
    'dqn_double_q': True,
    'dqn_network_hidden_size': 128,
    'dqn_grad_norm_clipping': 10,
    'logdir': None,
    'random_seed': 5,
    'save_params_freq': 50,
    'use_sqn': True,
    'sqn_alpha': 4
}
