import numpy as np
import torch
import os
from envs.gridworld import GridWorld
from algos.model import actorModel
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

DEVICE = "cuda:0"
EPISODES = 100000
STEPS = 1000
GAMMA = 0.99
RENDER = True
FRAMES = 1e8

env = GridWorld()
model = actorModel(4, 2).to(DEVICE)

model_dir = 'outputs/block/gridworld-vec-fix-reinforce-wprior_parallel0.1'
# model_dir = 'outputs/gridworld-vec-fix-manhattan-reinforce'
status = torch.load(model_dir + '/last_params_cloud.ckpt')
model.load_state_dict(status)

prior = torch.load(model_dir + '/prior_last.pt')

done = False
state = env.reset()
lp = []
r = []

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_xticks(np.arange(0, env.width+1, 1))
ax.set_yticks(np.arange(0, env.height+1, 1))
ax.set_aspect('equal')

plt.grid(True, axis='both', color='black', alpha=0.5, linestyle='--')
plt.xlim(0, env.width)
plt.ylim(0, env.height)

# agent
self_y, self_x = env.agent_pos
plt.fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')

# goal
goal_y, goal_x = env.goal
plt.fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')

goal = env.goal
for i in range(env.height):
    for j in range(env.width):
        agent_pos = np.array([i, j])
        state = agent_pos - goal
        actions = model(state)[0].tolist()

        self_y, self_x = i, j
        plt.arrow(self_x + 0.5, self_y + 0.5, -0.4 * actions[0], 0, facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        plt.arrow(self_x + 0.5, self_y + 0.5, 0.4 * actions[1], 0, facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        plt.arrow(self_x + 0.5, self_y + 0.5, 0, 0.4 * actions[2], facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        plt.arrow(self_x + 0.5, self_y + 0.5, 0, -0.4 * actions[3], facecolor='k', ec='k', head_width=0.1, head_length=0.1)

ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.flip(prior, axis=0))

plt.show()
