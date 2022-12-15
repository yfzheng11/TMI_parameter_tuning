import numpy as np
import random
import os
from infrastructure.dqn_utils import ReplayBuffer, PiecewiseSchedule
from argmax_policy import ArgMaxPolicy
from sqn_policy import SQNPolicy
from dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):
        self.env = env
        self.batch_size = agent_params['batch_size']

        self.num_actions = agent_params['ac_dim']
        self.num_patch_obs = int(agent_params['NPixel'] ** 2)

        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.exploration = PiecewiseSchedule([(0, 0.99), (1300, 0.1)], outside_value=0.1)
        # self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params)
        self.use_sqn = agent_params['use_sqn']
        self.actor = ArgMaxPolicy(self.critic)
        # if self.use_sqn:
        #     self.actor = SQNPolicy(self.critic)
        # else:
        #     self.actor = ArgMaxPolicy(self.critic)

        self.replay_buffer = ReplayBuffer(
            agent_params['replay_buffer_size'], int(agent_params['patch_obs'] ** 2))
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, state, action, param, reward, next_state, done):
        self.replay_buffer.store_sample(state, action, param, reward, next_state, done)

    def select_action_argmax_greedy(self):
        # use epsilon greedy exploration when selecting action
        eps = self.exploration.value(self.t)
        action = np.empty([self.num_patch_obs, self.env.NIMG], dtype=np.int32)

        for j in range(self.env.NIMG):
            # select action for each img in the env
            idx_lst = []
            for i in range(self.num_patch_obs):
                perform_random_action = (np.random.random() < eps) or (self.t < self.learning_starts)
                if perform_random_action:
                    # HINT: take random action (can sample from self.env.action_space)
                    # with probability eps (see np.random.random())
                    # OR if your current step number (see self.t) is less that self.learning_starts
                    action[i, j] = random.randint(0, self.num_actions - 1)
                else:
                    idx_lst.append(i)
                    # HINT: Your actor will take in multiple previous observations ("frames") in order
                    # to deal with the partial observability of the environment. Get the most recent
                    # `frame_history_len` observations using functionality from the replay buffer,
                    # and then use those observations as input to your actor.
                    # obs = self.replay_buffer.encode_recent_observation()
            action[idx_lst, j] = self.actor.get_action(self.env.obs[idx_lst, :, j])
        return action

    def select_action_policy(self):
        action = np.empty([self.num_patch_obs, self.env.NIMG], dtype=np.int32)

        for j in range(self.env.NIMG):
            action[:, j] = self.actor.get_action(self.env.obs[:, :, j])
        return action

    def step_env(self, done):
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

        # if self.use_sqn:
        #     actions = self.select_action_policy()
        # else:
        #     # use epsilon greedy exploration when selecting action
        #     actions = self.select_action_argmax_greedy()
        actions = self.select_action_argmax_greedy()

        # take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
        # obs, reward, done, info = env.step(action)
        last_obs = self.env.obs
        next_obs, params, reward, error, img_mat = self.env.step(actions)

        # store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        # self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        for i in range(self.env.NIMG):
            self.replay_buffer.store_sample(last_obs[:, :, i], actions[:, i], params[:, i],
                                            reward[:, i], next_obs[:, :, i], done[:, i])

        # if taking this step resulted in done, reset the env (and the latest observation)
        # if done:
        #     self.last_obs = self.env.reset()
        return error, np.mean(reward), img_mat

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

    def save(self, path):
        if not (os.path.exists(path)):
            os.makedirs(path)
        self.critic.save_models(path)
        self.env.save_env(path)
