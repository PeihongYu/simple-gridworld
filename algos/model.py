import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def init_params_2(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.orthogonal_(m.weight, nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 1e-6)


def to_tensor(obs, device):
    if len(obs.shape) == 1:
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    elif len(obs.shape) == 2:
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
    return obs


class ActorModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # layer_norm(self.layer1, nn.init.calculate_gain('relu'))
        # layer_norm(self.layer2, 0.1)

        # Initialize parameters correctly
        self.apply(init_params_2)

    def forward(self, obs):
        x = self.actor(to_tensor(obs, self.device))
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        return dist

    def get_action(self, dist):
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        return action, log_prob_action


class ACModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params_2)

    def forward(self, obs):
        obs = to_tensor(obs, self.device)
        x = self.actor(obs)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        y = self.critic(obs)
        value = y.squeeze(1)

        return dist, value

    def get_action(self, dist):
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        return action, log_prob_action
