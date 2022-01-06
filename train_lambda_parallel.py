import gym
import pandas as pd
import numpy as np
import torch
import os
import tensorboardX
from gridworld import GridWorld
from model import actorModel
from utils import calculate_lambda_parallel, calculate_lambda
from penv import ParallelEnv
import time

os.makedirs("outputs", exist_ok=True)

DEVICE = "cuda:0"
EPISODES = 60000
STEPS = 1000
GAMMA = 0.99
RENDER = False

env = GridWorld()

envs = []
for i in range(10):
    envs.append(GridWorld())
env_parallel = ParallelEnv(envs)

model = actorModel(4, 2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model_dir = 'outputs/gridworld-vec-fix-reinforce-wprior_parallel'
os.makedirs(model_dir, exist_ok=True)
tb_writer = tensorboardX.SummaryWriter(model_dir)

prior_dir = 'outputs/gridworld-vec-fix-reinforce'
prior_lambdas = torch.load(prior_dir + '/prior.pt')
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
    stime = time.time()
    lambdas = calculate_lambda_parallel(env_parallel, model)
    # lambdas = calculate_lambda(env, model)
    etime = time.time()
    print(etime - stime)

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
        prob1 = lambdas[i, j] * torch.exp(log_prob)

        actions = prior_model(state["vec"])
        p_prob = actions[0][action]

        prob2 = prior_lambdas[i, j] * p_prob
        shadow_r_ = torch.log(2 * prob1) - torch.log(prob1 + prob2)

        shadow_r.append(shadow_r_.item())

        state, r_, done, i_ = env.step(action)
        r.append(r_)
        step += 1

        if done or step == STEPS:
            state = env.reset()
            all_rewards.append(np.sum(r))
            all_shadow_rewards.append(np.sum(shadow_r))
            all_eplens.append(step)
            step = 0
            print(f"EPISODE {episode} STEP: {pd.Series(all_eplens).tail(100).mean()} SCORE: {np.sum(r)} roll: {pd.Series(all_rewards).tail(100).mean()}")
            tb_writer.add_scalar("frames", pd.Series(all_eplens).tail(100).mean(), episode)
            tb_writer.add_scalar("return", pd.Series(all_rewards).tail(100).mean(), episode)
            tb_writer.add_scalar("shadow return", pd.Series(all_shadow_rewards).tail(100).mean(), episode)

            if episode % 100 == 0:
                torch.save(model.state_dict(), model_dir + '/last_params_cloud.ckpt')
                if pd.Series(all_rewards).tail(100).mean() > best_rolling:
                    best_rolling = pd.Series(all_rewards).tail(100).mean()
                    print("saving...")
                    torch.save(model.state_dict(), model_dir + '/best_params_cloud.ckpt')
            break

    discounted_rewards = []

    r = np.array(r) - pweight * np.array(shadow_r)
    r = r.tolist()

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
