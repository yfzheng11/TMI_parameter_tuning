"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import random
from collections import namedtuple

import gym
import numpy as np
from torch import nn
import torch.optim as optim

from cs285.infrastructure.atari_wrappers import wrap_deepmind
from gym.envs.registration import register

import torch


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


OptimizerSpec = namedtuple(
    "OptimizerSpec",
    ["constructor", "optim_kwargs", "learning_rate_schedule"],
)


def register_custom_envs():
    from gym.envs.registration import registry
    if 'LunarLander-v3' not in registry.env_specs:
        register(
            id='LunarLander-v3',
            entry_point='cs285.envs.box2d.lunar_lander:LunarLander',
            max_episode_steps=1000,
            reward_threshold=200,
        )


def get_env_kwargs(env_name):
    if env_name in ['MsPacman-v0', 'PongNoFrameskip-v4']:
        kwargs = {
            'learning_starts': 50000,
            'target_update_freq': 10000,
            'replay_buffer_size': int(1e6),
            'num_timesteps': int(2e8),
            'q_func': create_atari_q_network,
            'learning_freq': 4,
            'grad_norm_clipping': 10,
            'input_shape': (84, 84, 4),
            'env_wrappers': wrap_deepmind,
            'frame_history_len': 4,
            'gamma': 0.99,
        }
        kwargs['optimizer_spec'] = atari_optimizer(kwargs['num_timesteps'])
        kwargs['exploration_schedule'] = atari_exploration_schedule(kwargs['num_timesteps'])

    elif env_name == 'LunarLander-v3':
        def lunar_empty_wrapper(env):
            return env

        kwargs = {
            'optimizer_spec': lander_optimizer(),
            'q_func': create_lander_q_network,
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 1000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 3000,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': 500000,
            'env_wrappers': lunar_empty_wrapper
        }
        kwargs['exploration_schedule'] = lander_exploration_schedule(kwargs['num_timesteps'])

    else:
        raise NotImplementedError

    return kwargs


def create_lander_q_network(ob_dim, num_actions):
    return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )


def create_lander_q_network_new(ob_dim, num_actions, num_hidden=5):
    assert num_hidden >= 2, 'need more hidden layers'
    layers = [nn.Linear(ob_dim, 64), nn.ReLU()]
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(64, 64))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(64, num_actions))
    return nn.Sequential(*layers)


class Ipdb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        import ipdb;
        ipdb.set_trace()
        return x


class PreprocessAtari(nn.Module):
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        return x / 255.


def create_atari_q_network(ob_dim, num_actions):
    return nn.Sequential(
        PreprocessAtari(),
        nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
        nn.ReLU(),
        nn.Linear(512, num_actions),
    )


def atari_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_ram_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_optimizer(num_timesteps):
    lr_schedule = PiecewiseSchedule(
        [
            (0, 1e-1),
            (num_timesteps / 40, 1e-1),
            (num_timesteps / 8, 5e-2),
        ],
        outside_value=5e-2,
    )

    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1e-3,
            eps=1e-4
        ),
        learning_rate_schedule=lambda t: lr_schedule.value(t),
    )


def lander_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=lambda epoch: 1e-3,  # keep init learning rate
    )


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables
    create ops corresponding to their exponential
    averages
    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.
    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happen if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left = new_vars_left


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


class ReplayBuffer(object):
    def __init__(self, size, ob_dim_squared):
        """
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        ob_dim_squared: int
            dimention for each observation.
        """

        self.size = size
        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = np.empty((self.size, ob_dim_squared), dtype=np.float32)
        self.next_obs = np.empty((self.size, ob_dim_squared), dtype=np.float32)
        self.param = np.empty([self.size], dtype=np.float32)
        self.action = np.empty([self.size], dtype=np.int32)
        self.reward = np.empty([self.size], dtype=np.float32)
        self.done = np.empty([self.size], dtype=np.bool)

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)

        obs_batch = self.obs[idxes, :]
        act_batch = self.action[idxes]
        param_batch = self.param[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = self.next_obs[idxes, :]
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, param_batch, rew_batch, next_obs_batch, done_mask

    def store_sample(self, state, action, param, reward, next_state, done):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        state: np.array
            Array of shape (n, img_h * img_w, img_c) and dtype np.uint8
            the frame to be stored
        next_state: np.array,
            Array of shape (1, img_h * img_w * img_c) and dtype np.uint8
            the frame to be stored
        action: int
            Action that was performed upon observing this frame.
        param: float
            param for this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        num = len(action)
        if num + self.next_idx - 1 <= self.size:
            self.obs[self.next_idx:num + self.next_idx, :] = state
            self.next_obs[self.next_idx:num + self.next_idx, :] = next_state
            self.action[self.next_idx:num + self.next_idx] = action
            self.param[self.next_idx:num + self.next_idx] = param
            self.reward[self.next_idx:num + self.next_idx] = reward
            self.done[self.next_idx:num + self.next_idx] = done
            # update next idx and total samples in the buffer
            self.next_idx = (self.next_idx + num) % self.size
            self.num_in_buffer = min(self.size, self.num_in_buffer + num)
        else:
            part1 = self.size - self.next_idx + 2
            part2 = num - self.size + self.next_idx - 1
