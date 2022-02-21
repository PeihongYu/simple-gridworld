import numpy as np
import matplotlib.pyplot as plt

def get_policy(agent_pos):
    # left, right, up, down
    ay, ax = (agent_pos + 9)
    if ax >= 9:
        return [0, 0, 1, 0]
    return [0, 1, 0, 0]
    # if ax >= 8:
    #     return [0, 1, 0, 0]
    # if ay <= 1:
    #     return [0, 1, 0, 0]
    # return [0.25, 0.25, 0.25, 0.25]


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_xticks(np.arange(0, 10+1, 1))
ax.set_yticks(np.arange(0, 10+1, 1))
ax.set_aspect('equal')

plt.grid(True, axis='both', color='black', alpha=0.5, linestyle='--')
plt.xlim(0, 10)
plt.ylim(0, 10)

# agent
self_y, self_x = 0, 0
plt.fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')

# goal
goal_y, goal_x = 9, 9
plt.fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')

goal = [9, 9]
for i in range(10):
    for j in range(10):
        agent_pos = np.array([i, j])
        state = agent_pos - goal
        actions = get_policy(state)

        self_y, self_x = i, j
        plt.arrow(self_x + 0.5, self_y + 0.5, -0.4 * actions[0], 0, facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        plt.arrow(self_x + 0.5, self_y + 0.5, 0.4 * actions[1], 0, facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        plt.arrow(self_x + 0.5, self_y + 0.5, 0, 0.4 * actions[2], facecolor='k', ec='k', head_width=0.1, head_length=0.1)
        plt.arrow(self_x + 0.5, self_y + 0.5, 0, -0.4 * actions[3], facecolor='k', ec='k', head_width=0.1, head_length=0.1)

ax = fig.add_subplot(1, 2, 2)


a = np.zeros([10, 10])

for i in range(10):
    for j in range(10):
        if i > 1 and j < 8:
            a[i, j] = 0
        else:
            a[i, j] = 1 / 36
print(a.sum())


plt.imshow(np.flip(a, axis=0))

plt.show()
