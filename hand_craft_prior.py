import numpy as np
import json
from enum import IntEnum
np.set_printoptions(precision=2)


class Actions(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3
    stay = 4


# upperLeftSquare_1a
# centerSquare_1a
# centerSquare_2a
# empty_1a
env_name = "centerSquare_1a"
json_file = "./envfiles/" + env_name + ".json"

with open(json_file) as infile:
    args = json.load(infile)
grid = np.load(args["grid_file"])
reward_mat = np.load(args["reward_file"])
height, width = grid.shape
agent_num = args["agent_num"]
state_lambdas = np.zeros(grid.shape)
sa_lambdas = np.zeros((5,) + grid.shape)


if env_name == "centerSquare_1a":
    num_visit = (reward_mat == 0).sum()
    state_lambdas[reward_mat == 0] = 1 / num_visit
    i = 0
    for j in range(1, width-1):
        sa_lambdas[Actions.right, i, j] = state_lambdas[i, j]
    j = 0
    for i in range(1, height-1):
        sa_lambdas[Actions.up, i, j] = state_lambdas[i, j]
    i = j = 0
    sa_lambdas[Actions.right, i, j] = state_lambdas[i, j] * 0.5
    sa_lambdas[Actions.up, i, j] = state_lambdas[i, j] * 0.5
    i = 9
    for j in range(width-1):
        sa_lambdas[Actions.right, i, j] = state_lambdas[i, j]
    j = 9
    for i in range(height-1):
        sa_lambdas[Actions.up, i, j] = state_lambdas[i, j]
    i = j = 9
    sa_lambdas[:, i, j] = state_lambdas[i, j] * 0.25

if env_name == "centerSquare_1a_flip":
    num_visit = (reward_mat == 0).sum()
    state_lambdas[reward_mat == 0] = 1 / num_visit
    i = 9
    for j in range(1, width - 1):
        sa_lambdas[Actions.left, i, j] = state_lambdas[i, j]
    j = 9
    for i in range(1, height - 1):
        sa_lambdas[Actions.down, i, j] = state_lambdas[i, j]
    i = j = 9
    sa_lambdas[Actions.right, i, j] = state_lambdas[i, j] * 0.5
    sa_lambdas[Actions.up, i, j] = state_lambdas[i, j] * 0.5
    i = 0
    for j in range(1, width):
        sa_lambdas[Actions.left, i, j] = state_lambdas[i, j]
    j = 0
    for i in range(1, height):
        sa_lambdas[Actions.down, i, j] = state_lambdas[i, j]
    i = j = 0
    sa_lambdas[:, i, j] = state_lambdas[i, j] * 0.25

print(state_lambdas.sum())
print(sa_lambdas.sum())
print(sa_lambdas.sum(0))

np.save("./envfiles/" + env_name + "_prior.npy", sa_lambdas)