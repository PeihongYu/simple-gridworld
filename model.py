import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda:0"
ACTION_SPACE = [0, 1, 2, 3]


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.manual_seed(786)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class actorModel(nn.Module):
    def __init__(self, num_action, num_input):
        super().__init__()
        self.num_action = num_action
        self.num_input = num_input

        self.layer1 = nn.Linear(num_input, 64)
        self.layer2 = nn.Linear(64, num_action)

        layer_norm(self.layer1, nn.init.calculate_gain('relu'))
        layer_norm(self.layer2, 0.1)
        # print(self.layer1.weight.tolist(), self.layer1.weight.tolist())
        # print(self.layer2.weight.tolist(), self.layer2.weight.tolist())
        # print("done")

    def forward(self, x):
        if len(x.shape) == 1:
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        elif len(x.shape) ==2:
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.layer1(x))
        actions = F.softmax(self.layer2(x), dim=1)
        return actions

    def get_action(self, actions):
        action = np.random.choice(ACTION_SPACE, p=actions.squeeze(0).detach().cpu().numpy())
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action, log_prob_action
