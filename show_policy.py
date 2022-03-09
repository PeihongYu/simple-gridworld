import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from envs.gridworld import GridWorldEnv
import utils

# upperLeftSquare_1a
# centerSquare_1a
# centerSquare_2a
# empty_1a
env_name = "centerSquare_2a"
json_file = "./envfiles/" + env_name + ".json"
env = GridWorldEnv(json_file, True)
agent_num = env.agent_num

algorithm = "PPO"

model_dir = 'outputs/centerSquare_2a_PPO_wprior0.995'

acmodels, select_action = utils.load_models(model_dir, env)


fig, axs = plt.subplots(1, agent_num)
for i in range(agent_num):
    self_y, self_x = env.agents[i]
    axs[i].fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')
    goal_y, goal_x = env.goals[i]
    axs[i].fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')

    axs[i].set_xticks(np.arange(0, env.width + 1, 1))
    axs[i].set_yticks(np.arange(0, env.height + 1, 1))
    axs[i].set_aspect('equal')

    axs[i].grid(True, axis='both', color='black', alpha=0.5, linestyle='--')
    axs[i].set_xlim(0, env.width-1)
    axs[i].set_xlim(0, env.height)
    axs[i].set_title("policy of agent " + str(i))

for i in range(env.height):
    for j in range(env.width):
        agent_pos = np.array([i, j])
        state = agent_pos - goal
        actions = model(state)[0].tolist()

        self_y, self_x = i, j
        ax.arrow(self_x + 0.5, self_y + 0.5, -0.4 * actions[0], 0, facecolor='k', ec='k', head_width=0.1,
                  head_length=0.1)
        ax.arrow(self_x + 0.5, self_y + 0.5, 0.4 * actions[1], 0, facecolor='k', ec='k', head_width=0.1,
                  head_length=0.1)
        ax.arrow(self_x + 0.5, self_y + 0.5, 0, 0.4 * actions[2], facecolor='k', ec='k', head_width=0.1,
                  head_length=0.1)
        ax.arrow(self_x + 0.5, self_y + 0.5, 0, -0.4 * actions[3], facecolor='k', ec='k', head_width=0.1,
                  head_length=0.1)


plt.show()
