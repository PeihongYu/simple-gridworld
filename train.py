import gym
import pandas as pd
import numpy as np
import torch
import os
import tensorboardX
from gridworld import GridWorld
from model import actorModel
from tqdm import tqdm
import math


os.makedirs("outputs", exist_ok=True)

DEVICE = "cuda:0"
EPISODES = 10000
STEPS = 1000
GAMMA = 0.99
RENDER = False

env = GridWorld()
model = actorModel(4, 2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# model_dir = 'outputs/gridworld-vec-fix-reinforce-wprior'
model_dir = 'outputs/gridworld-vec-fix-reinforce2'
os.makedirs(model_dir, exist_ok=True)
tb_writer = tensorboardX.SummaryWriter(model_dir)

num_frames = 0
all_rewards = []
all_eplens = []
best_rolling = -99999

for episode in range(EPISODES):

    done = False
    step = 0
    state = env.reset()
    lp = []
    r = []

    while not done:
        if RENDER:
            env.render()
        actions = model(state["vec"])
        action, log_prob = model.get_action(actions)
        lp.append(log_prob)

        i, j = env.agent_pos

        state, r_, done, i_ = env.step(action)
        r.append(r_)
        step += 1

        if done or step == STEPS:
            state = env.reset()
            all_rewards.append(np.sum(r))
            all_eplens.append(step)
            step = 0
            print(
                f"EPISODE {episode} STEP: {pd.Series(all_eplens).tail(100).mean()} SCORE: {np.sum(r)} roll: {pd.Series(all_rewards).tail(100).mean()}")
            tb_writer.add_scalar("frames", pd.Series(all_eplens).tail(100).mean(), episode)
            tb_writer.add_scalar("return", pd.Series(all_rewards).tail(100).mean(), episode)

            if episode % 100 == 0:
                torch.save(model.state_dict(), model_dir + '/last_params_cloud.ckpt')
                if pd.Series(all_rewards).tail(100).mean() > best_rolling:
                    best_rolling = pd.Series(all_rewards).tail(100).mean()
                    print("saving...")
                    torch.save(model.state_dict(), model_dir + '/best_params_cloud.ckpt')
            break

    discounted_rewards = []

    for t in range(len(r)):
        Gt = 0
        pw = 0
        for r_ in r[t:]:
            Gt = Gt + GAMMA ** pw * r_
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = np.array(discounted_rewards)

    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
    discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
    log_prob = torch.stack(lp)
    policy_gradient = -log_prob * discounted_rewards

    model.zero_grad()
    policy_gradient.sum().backward()
    optimizer.step()
