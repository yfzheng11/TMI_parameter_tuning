import numpy as np
import torch


class SQNPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        with torch.no_grad():
            qa_value = self.critic.qa_values(obs)
            v = self.critic.get_V_for_sqn(qa_value)
            pi_maxent = torch.exp((qa_value - v) / self.critic.alpha)
            pi_maxent = pi_maxent / pi_maxent.sum(dim=-1, keepdim=True)
            dist = torch.distributions.Categorical(pi_maxent)
            action = dist.sample().item()
        return action
