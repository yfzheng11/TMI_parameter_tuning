import numpy as np
import random
from infrastructure.dqn_utils import ReplayBuffer, PiecewiseSchedule
from argmax_policy import ArgMaxPolicy
from dqn_critic import DQNCritic
import mmlem_recon as emrecon


class DQNAgent(object):
    def __init__(self, agent_params):
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.sysmat = agent_params['sysmat']
        # import ipdb; ipdb.set_trace()
        # self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.num_patches = agent_params['num_patch']
        self.num_pixels = agent_params['num_pixel']
        self.patch_size = agent_params['patch_size']
        self.patch_rew = agent_params['patch_rew']
        self.niter = agent_params['niter']

        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        # self.replay_buffer_idx = None
        self.exploration = PiecewiseSchedule([(0, 0.99), (1300, 0.1)], outside_value=0.1)
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params['ob_dim'], agent_params['ac_dim'])
        self.actor = ArgMaxPolicy(self.critic)

        # lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = ReplayBuffer(
            agent_params['replay_buffer_size'], int(agent_params['ob_dim'] ** 2))
        self.t = 0
        self.num_param_updates = 0

        self._int_to_action = {
            '0': 1.5,
            '1': 1.1,
            '2': 1.0,
            '3': 0.9,
            '4': 0.5,
        }

    def add_to_replay_buffer(self, state, action, param, reward, next_state, done):
        self.replay_buffer.store_sample(state, action, param, reward, next_state, done)

    def update_param(self, old_param, actions):
        param = np.empty([self.num_patches], dtype=np.float32)
        for i in range(self.num_actions):
            param[actions == i] = old_param[actions == i] * self._int_to_action[f'{i}']
        return param

    def step_env(self, obs, proj, old_param, ground_truth, done):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """

        # store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
        # in dqn_utils.py
        # self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        # use epsilon greedy exploration when selecting action
        eps = self.exploration.value(self.t)
        action = np.empty([self.num_patches], dtype=np.int32)

        idx_lst = []
        for i in range(self.num_patches):
            perform_random_action = (np.random.random() < eps) or (self.t < self.learning_starts)
            if perform_random_action:
                # HINT: take random action (can sample from self.env.action_space)
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
                action[i] = random.randint(0, self.num_actions - 1)
            else:
                idx_lst.append(i)
                # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor.
                # obs = self.replay_buffer.encode_recent_observation()
        action[idx_lst] = self.actor.get_action(obs[idx_lst, :])
        param = self.update_param(old_param, action)

        # take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
        # obs, reward, done, info = env.step(action)
        next_obs, reward, error, fimg = emrecon.mlem_tv(self.sysmat, proj, obs, param, ground_truth, self.num_pixels,
                                                        self.patch_size, self.patch_rew, self.niter)
        # self.last_obs = new_obs

        # store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        # self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        self.replay_buffer.store_sample(obs, action, param, reward, next_obs, done)

        # if taking this step resulted in done, reset the env (and the latest observation)
        # if done:
        #     self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], []

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)):

            # fill in the call to the update function using the appropriate tensors
            log = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )

            # update the target network periodically
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
