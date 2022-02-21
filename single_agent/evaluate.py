import torch
import os
from envs.gridworld import GridWorld
from model import actorModel

os.makedirs("outputs", exist_ok=True)


DEVICE = "cuda:0"
EPISODES = 100000
STEPS = 1000
GAMMA = 0.99
RENDER = True
FRAMES = 1e8


env = GridWorld()
model = actorModel(4, 2).to(DEVICE)

model_dir = 'outputs/block/gridworld-vec-fix-reinforce'
# model_dir = 'outputs/gridworld-vec-fix-manhattan-reinforce'
status = torch.load(model_dir + '/best_params_cloud.ckpt')
model.load_state_dict(status)

for episode in range(EPISODES):
    done = False
    state = env.reset()
    lp = []
    r = []

    for step in range(STEPS):
        if RENDER:
            env.render()
        actions = model(state["vec"])
        action, log_prob = model.get_action(actions)
        state, r_, done, i_ = env.step(action)
        lp.append(log_prob)
        r.append(r_)

        if done:
            state = env.reset()
            break

