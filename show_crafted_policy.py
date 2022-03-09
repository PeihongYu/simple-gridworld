import numpy as np
import matplotlib.pyplot as plt
from envs.gridworld import GridWorldEnv

# centerSquare_1a
# centerSquare_1a_flip
env_name = "centerSquare_4a"
aid = 2
json_file = "./envfiles/" + env_name + ".json"
env = GridWorldEnv(json_file)
prior = np.load("./envfiles/" + env_name + "_prior" + str(aid) + ".npy")

state_prior = prior.sum(0)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
ax.set_xticks(np.arange(0, 10+1, 1))
ax.set_yticks(np.arange(0, 10+1, 1))
ax.set_aspect('equal')

plt.grid(True, axis='both', color='black', alpha=0.5, linestyle='--')
plt.xlim(0, 10)
plt.ylim(0, 10)

# agent
self_y, self_x = env.agents[aid]
plt.fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')

# goal
goal_y, goal_x = env.goals[aid]
plt.fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')


for i in range(10):
    for j in range(10):
        actions = prior[:, i, j]
        if prior[:, i, j].sum() > 0:
            actions /= prior[:, i, j].sum()

        self_y, self_x = i, j
        if actions[0] > 0:
            plt.arrow(self_x + 0.5, self_y + 0.5, -0.4 * actions[0], 0, facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        if actions[1] > 0:
            plt.arrow(self_x + 0.5, self_y + 0.5, 0.4 * actions[1], 0, facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        if actions[2] > 0:
            plt.arrow(self_x + 0.5, self_y + 0.5, 0, 0.4 * actions[2], facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        if actions[3] > 0:
            plt.arrow(self_x + 0.5, self_y + 0.5, 0, -0.4 * actions[3], facecolor='k', ec='k', head_width=0.1, head_length=0.1)

ax = fig.add_subplot(1, 2, 2)

plt.imshow(np.flip(state_prior, axis=0))

# plt.show()
plt.savefig("./envfiles/" + env_name + "_prior" + str(aid) + '.jpg')
plt.close(fig)
