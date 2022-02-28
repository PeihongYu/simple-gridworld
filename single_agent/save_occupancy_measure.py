import numpy as np
import torch
import os
from envs.gridworld import GridWorld
from algos.model import actorModel

os.makedirs("outputs", exist_ok=True)


DEVICE = "cuda:0"
EPISODES = 10000
STEPS = 1000
GAMMA = 0.99
RENDER = False
FRAMES = 1e8


env = GridWorld()
model = actorModel(4, 2).to(DEVICE)

model_dir = 'outputs/block/gridworld-vec-fix-reinforce-wprior_parallel0.1'
status = torch.load(model_dir + '/last_params_cloud.ckpt')
model.load_state_dict(status)

lambdas = np.zeros([env.height, env.width])

for episode in range(EPISODES):
    print("episode: ", episode)
    done = False
    state = env.reset()

    local_lambdas = np.zeros([env.height, env.width])
    i, j = env.agent_pos
    local_lambdas[i, j] += 1

    while not done:
        if RENDER:
            env.render()
        actions = model(state["vec"])
        action, log_prob = model.get_action(actions)
        state, r_, done, i_ = env.step(action)

        i, j = env.agent_pos
        local_lambdas[i, j] += 1

        if done:
            state = env.reset()
            break

    lambdas += local_lambdas / local_lambdas.sum()

lambdas /= lambdas.sum()

torch.save(lambdas, model_dir + '/prior.pt')
