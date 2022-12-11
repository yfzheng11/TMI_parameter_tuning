from collections import OrderedDict
import pickle
import os
import sys
import time
from infrastructure.logger import Logger
from dqn_agent import DQNAgent

from cs285.infrastructure.atari_wrappers import ReturnWrapper

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure.utils import Path
from cs285.infrastructure import utils

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, agent, params):
        #############
        ## INIT
        #############
        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(use_gpu=True, gpu_id=0)

        #############
        ## AGENT
        #############
        self.agent = agent

    def run_training_loop(self, n_iter, collect_policy=None, initial_expertdata=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param initial_expertdata:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 10 if isinstance(self.agent, DQNAgent) else 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent):
                # only perform an env step and add to replay buffer for DQN
                self.agent.env.reset()
            else:
                use_batchsize = self.params['batch_size']
                if itr == 0:
                    use_batchsize = self.params['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policy, use_batchsize)
                )

            # add collected data to replay buffer
            # self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_dqn_logging(all_logs)
                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample,
                                      save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # get this from hw1 or hw2
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, num_transitions_to_sample,
                                                               self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.logvideo:
            print('\nCollecting train rollouts to be used for saving videos...')
            # look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        # get this from hw1 or hw2
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # step env
            error, img_mat = self.agent.step_env()
            self.total_envsteps += 1

            # sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, param_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['batch_size'])

            # use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = self.env.get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()
