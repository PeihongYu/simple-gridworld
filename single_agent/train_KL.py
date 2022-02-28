import pandas as pd
import numpy as np
import torch
import os
import tensorboardX
from envs.gridworld import GridWorld
from algos.model import actorModel


def get_policy(agent_pos):
    # left, right, up, down
    ay, ax = (agent_pos + 9)
    if ax >= 8:
        return [0, 0, 1, 0]
    if ay <= 1:
        return [0, 1, 0, 0]
    return [0.25, 0.25, 0.25, 0.25]


def KL_divergence(p, q):
    if 0 in p:
        p += 1e-9
        p /= p.sum()
    if 0 in q:
        q += 1e-9
        q /= q.sum()
    return sum(p[i] * torch.log2(p[i] / q[i]) for i in range(len(p)))


os.makedirs("outputs", exist_ok=True)

DEVICE = "cuda:0"
EPISODES = 10000
STEPS = 1000
GAMMA = 0.99
RENDER = False
use_prior = True

env = GridWorld()
model = actorModel(4, 2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model_dir = 'outputs/block/gridworld-vec-fix-reinforce-wpriorkl0.1'
os.makedirs(model_dir, exist_ok=True)
tb_writer = tensorboardX.SummaryWriter(model_dir)

if use_prior:
    prior_dir = 'outputs/sparse/gridworld-vec-fix-reinforce'
    prior_model = actorModel(4, 2).to(DEVICE)
    status = torch.load(prior_dir + '/best_params_cloud.ckpt')
    prior_model.load_state_dict(status)

num_frames = 0
all_rewards = []
all_shadow_rewards = []
all_eplens = []
best_rolling = -99999
pweight = 0.1

for episode in range(EPISODES):

    done = False
    step = 0
    state = env.reset()
    lp = []
    r = []
    shadow_r = []

    while not done:
        if RENDER:
            env.render()
        actions = model(state["vec"])
        action, log_prob = model.get_action(actions)
        lp.append(log_prob)

        i, j = env.agent_pos
        if use_prior:
            # prior_dist = prior_model(state["vec"])[0]
            prior_dist = torch.tensor(get_policy(state["vec"]), dtype=torch.float32, device=DEVICE)
            shadow_r.append(KL_divergence(actions[0], prior_dist))

        state, r_, done, i_ = env.step(action)
        r.append(r_)
        step += 1

        if done or step == STEPS:
            state = env.reset()
            all_rewards.append(np.sum(r))
            all_shadow_rewards.append(torch.stack(shadow_r).sum().item())
            all_eplens.append(step)
            step = 0
            print(
                f"EPISODE {episode} STEP: {pd.Series(all_eplens).tail(100).mean()} SCORE: {np.sum(r)} roll: {pd.Series(all_rewards).tail(100).mean()}")
            tb_writer.add_scalar("frames", pd.Series(all_eplens).tail(100).mean(), episode)
            tb_writer.add_scalar("return", pd.Series(all_rewards).tail(100).mean(), episode)
            tb_writer.add_scalar("KL", pd.Series(all_shadow_rewards).tail(100).mean(), episode)

            if episode % 100 == 0:
                torch.save(model.state_dict(), model_dir + '/last_params_cloud.ckpt')
                if pd.Series(all_rewards).tail(100).mean() > best_rolling:
                    best_rolling = pd.Series(all_rewards).tail(100).mean()
                    print("saving...")
                    torch.save(model.state_dict(), model_dir + '/best_params_cloud.ckpt')
            break

    discounted_rewards = []

    if use_prior:
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE) - pweight * torch.stack(shadow_r)
        # r = -torch.stack(shadow_r)

    for t in range(len(r)):
        Gt = 0
        pw = 0
        for r_ in r[t:]:
            Gt = Gt + GAMMA ** pw * r_
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.stack(discounted_rewards)
    discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
    log_prob = torch.stack(lp)
    policy_gradient = -log_prob * discounted_rewards

    model.zero_grad()
    policy_gradient.sum().backward()
    optimizer.step()
