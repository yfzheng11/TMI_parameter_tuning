import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
from infrastructure import pytorch_util as ptu
import copy


class DQNCritic(object):

    def __init__(self, critic_params):
        self.ob_dim = critic_params['patch_obs']
        self.ac_dim = critic_params['ac_dim']
        self.gamma = critic_params['dqn_discount_rate_gamma']
        self.learning_rate = critic_params['dqn_learning_rate']

        self.double_q = critic_params['dqn_double_q']
        self.grad_norm_clipping = critic_params['dqn_grad_norm_clipping']

        self.q_net = self.build_network(h_size=critic_params['dqn_network_hidden_size'])
        self.q_net_target = self.build_network(h_size=critic_params['dqn_network_hidden_size'])

        self.optimizer = optim.Adam(self.q_net.parameters(),
                                    self.learning_rate)
        self.learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200],
                                                                      gamma=0.5)
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

    def build_network(self, h_size=128):
        return nn.Sequential(
            nn.Unflatten(1, (1, self.ob_dim, self.ob_dim)),
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3 * 3 * 64, h_size),
            nn.ReLU(inplace=True),
            nn.Linear(h_size, self.ac_dim),
        )

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)

        # compute the Q-values from the target network
        qa_tp1_values = self.q_net_target(next_ob_no)

        if self.double_q:
            # You must fill this part for Q2 of the Q-learning portion of the homework.
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network. Please review Lecture 8 for more details,
            # and page 4 of https://arxiv.org/pdf/1509.06461.pdf is also a good reference.
            ac_next = self.q_net(next_ob_no).argmax(dim=1)
            q_tp1 = torch.gather(qa_tp1_values, 1, ac_next.unsqueeze(1)).squeeze(1)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)

        # compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        """Returns Q(s, a)
                Args:
                    obs (np.ndarray): State array, shape (n, ob_dim)

                Returns:
                    np.ndarray: Q value array, shape (n, ac_dim)
        """
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)

    def save_models(self, path):
        print("saving model weights")
        torch.save(self.q_net.state_dict(), f'{path}/qnet_wts.mdl')
        torch.save(self.q_net_target.state_dict(), f'{path}/qtarget_wts.mdl')
