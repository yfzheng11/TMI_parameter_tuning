import numpy as np
import torch
from infrastructure import pytorch_util as ptu


class SQNPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        with torch.no_grad():
            obs = ptu.from_numpy(obs)
            qa_value = self.critic.q_net(obs)
            v = self.critic.get_V_for_sqn(qa_value)
            pi_maxent = torch.exp((qa_value - v) / self.critic.alpha)
            pi_maxent = pi_maxent / pi_maxent.sum(dim=-1, keepdim=True)
            distribution = torch.distributions.Categorical(pi_maxent)
            action = distribution.sample()
        return ptu.to_numpy(action)
