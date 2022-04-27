import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from envs.gridworld import GridWorldEnv
import utils

env_name = "centerSquare6x6_1a"
env = GridWorldEnv(env_name, True)
agent_num = env.agent_num

ACTIONS_NAME = ["left", "right", "up", "down", "stay"]

algorithm = "pofd"

# model_dir = 'outputs/centerSquare6x6_3a_PPO_wprior0.995_N1000_gradNoise'
# model_dir = 'outputs_lava/centerSquare_1a_PPO'
model_dir = 'outputs_lava_batch/centerSquare6x6_1a_POfD_r0_pw1.0_pd1.0'

discriminators = utils.load_discriminator(model_dir, env, "best")

aid = 0
fig, axs = plt.subplots(2, 2)
rewards = np.zeros([env.action_space.n, env.height, env.width])

for i in range(2):
    for j in range(2):
        self_y, self_x = env.agents[aid]
        axs[i][j].fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')
        goal_y, goal_x = env.goals[aid]
        axs[i][j].fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')

        axs[i][j].set_xticks(np.arange(0, env.width + 1, 1))
        axs[i][j].set_yticks(np.arange(0, env.height + 1, 1))
        axs[i][j].set_aspect('equal')

        axs[i][j].grid(True, axis='both', color='black', alpha=0.5, linestyle='--')
        axs[i][j].set_xlim(0, env.width-1)
        axs[i][j].set_xlim(0, env.height)

        action = i * 2 + j
        for x in range(env.height):
            for y in range(env.width):
                if 1 < x < 8 and 1 < y < 8:
                    continue
                state = np.array([x, y])
                actions = np.eye(env.action_space.n)[action]
                reward = -torch.log(discriminators[aid](np.hstack([state, actions])))
                # reward = discriminators[aid](np.hstack([state, actions]))
                rewards[action, x, y] = reward
                axs[i][j].text(x+0.1, y+0.5, round(reward.item(), 3))

        axs[i][j].set_title("reward of action " + ACTIONS_NAME[action])

rbest = np.argmax(rewards, axis=0)
for x in range(env.height):
    for y in range(env.width):
        i = rbest[x, y] // 2
        j = rbest[x, y] % 2
        axs[i][j].fill([x, x, x + 1, x + 1], [y, y + 1, y + 1, y], facecolor='pink')


plt.show()
