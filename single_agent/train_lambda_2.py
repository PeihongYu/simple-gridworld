import pandas as pd
import numpy as np
import torch
import os
import tensorboardX
from envs.gridworld import GridWorld
from algos.model import actorModel
from lambda_helper import calculate_lambda
from utils.penv import ParallelEnv
import time
import matplotlib.pyplot as plt


np.set_printoptions(precision=2)


def get_policy(agent_pos):
    # left, right, up, down
    ay, ax = (agent_pos + 9)
    if ax >= 9:
        return [0, 0, 1, 0]
    if ax >= 8:
        return [0, 1, 0, 0]
    if ay <= 1:
        return [0, 1, 0, 0]
    return [0.25, 0.25, 0.25, 0.25]


def get_location(ax, ay, action):
    if action == 0:
        return [ax, ay + 0.45], [[ax, ay], [ax, ay+1], [ax+0.5, ay+0.5]]
    if action == 1:
        return [ax + 0.55, ay + 0.45], [[ax+1, ay], [ax+1, ay+1], [ax+0.5, ay+0.5]]
    if action == 2:
        return [ax + 0.375, ay + 0.8], [[ax, ay+1], [ax+1, ay+1], [ax+0.5, ay+0.5]]
    if action == 3:
        return [ax + 0.375, ay], [[ax, ay], [ax+1, ay], [ax+0.5, ay+0.5]]


def get_alpha(x, minx, maxx):
    if x > 0:
        return x / maxx
    else:
        return x / minx
    # return (x - minx) / (maxx - minx)


def get_color(x):
    if x > 0:
        return "blue"
    else:
        return "red"


os.makedirs("outputs", exist_ok=True)

DEVICE = "cuda:0"
EPISODES = 10000
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

model_dir = 'outputs/block/gridworld-vec-fix-reinforce-wprior-only'
os.makedirs(model_dir, exist_ok=True)
img_dir = model_dir + '/imgs'
os.makedirs(img_dir, exist_ok=True)


try:
    status = torch.load(model_dir + '/last_params_cloud.pt', map_location=DEVICE)
    # local model and optimizer
    model.load_state_dict(status["model_state"])
    optimizer.load_state_dict(status["optimizer_state"])
    # local logging information
    episode = status["episode"]
    all_rewards = status["all_rewards"]
    all_shadow_rewards = status["all_shadow_rewards"]
    all_eplens = status["all_eplens"]
    best_rolling = status["best_rolling"]

except OSError:
    episode = 0
    all_rewards = []
    all_shadow_rewards = []
    all_eplens = []
    best_rolling = -99999

tb_writer = tensorboardX.SummaryWriter(model_dir)

# prior_dir = 'outputs/gridworld-vec-fix-reinforce'
# prior_lambdas = torch.load(prior_dir + '/prior.pt')
# prior_model = actorModel(4, 2).to(DEVICE)
# status = torch.load(prior_dir + '/best_params_cloud.ckpt')
# prior_model.load_state_dict(status)

pweight = 0.1

while episode < EPISODES:
    stime = time.time()
    # lambdas = calculate_lambda_parallel(env_parallel, model, N=1000)
    lambdas = calculate_lambda(env, model, N=1000)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0, 0].imshow(np.flip(lambdas, axis=0))
    axs[0, 0].set_title("state occupancy measure")

    ids = [[0, 1], [1, 0], [1, 1]]
    titles = ["shadow rewards", "total return", "shadow return"]

    for i in range(3):
        axs[ids[i][0], ids[i][1]].set_xticks(np.arange(0, env.width + 1, 1))
        axs[ids[i][0], ids[i][1]].set_yticks(np.arange(0, env.height + 1, 1))
        axs[ids[i][0], ids[i][1]].set_aspect('equal')

        axs[ids[i][0], ids[i][1]].grid(True, axis='both', color='black', alpha=0.5, linestyle='--')
        axs[ids[i][0], ids[i][1]].set_xlim(0, env.width)
        axs[ids[i][0], ids[i][1]].set_xlim(0, env.height)
        axs[ids[i][0], ids[i][1]].set_title(titles[i])

    etime = time.time()
    print(etime - stime)

    done = False
    step = 0
    state = env.reset()
    lp = []
    r = []
    shadow_r = []
    agents_locs = []

    while not done:
        if RENDER:
            env.render()
        actions = model(state["vec"])
        action, log_prob = model.get_action(actions)
        lp.append(log_prob)

        i, j = env.agent_pos
        agents_locs.append([j, i, action])
        prob1 = lambdas[i, j] * torch.exp(log_prob)

        # actions = prior_model(state["vec"])
        # p_prob = actions[0][action]

        # prob2 = prior_lambdas[i, j] * p_prob

        if i > 1 and j < 8:
            prob2 = 1e-12
        else:
            p_probs = torch.tensor(get_policy(state["vec"]), dtype=torch.float32, device=DEVICE)
            p_prob = p_probs[action]
            prob2 = 1/36 * p_prob
            if prob2 == 0:
                prob2 == 1e-12

        shadow_r_ = torch.log(2 * prob1) - torch.log(prob1 + prob2)
        shadow_r.append(-shadow_r_.item())

        if i > 1 and j < 8 and shadow_r_ < 0:
            print(prob1)

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

            if episode % 10 == 0:
                status = {"episode": episode,
                          "model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(),
                          "all_rewards": all_rewards,
                          "all_shadow_rewards": all_shadow_rewards,
                          "all_eplens": all_eplens,
                          "best_rolling": best_rolling}

                torch.save(status, model_dir + '/last_params_cloud.pt')
                if pd.Series(all_rewards).tail(100).mean() > best_rolling:
                    best_rolling = pd.Series(all_rewards).tail(100).mean()
                    print("saving...")
                    torch.save(status, model_dir + '/best_params_cloud.pt')
            break

    discounted_rewards = []

    # r = np.array(r) + pweight * np.array(shadow_r)
    r = np.array(shadow_r)
    r = r.tolist()

    discounted_rewards = [0] * len(r)
    discounted_shadowrs = [0] * len(r)
    for t in reversed(range(len(r))):
        discounted_rewards[t] = r[t] + GAMMA * discounted_rewards[t + 1] if t < len(r)-1 else r[t]
        discounted_shadowrs[t] = shadow_r[t] + GAMMA * discounted_shadowrs[t + 1] if t < len(r) - 1 else r[t]


    min_shadow_r = min(shadow_r)
    max_shadow_r = max(shadow_r)
    min_dis_rewards = min(discounted_rewards)
    max_dis_rewards = max(discounted_rewards)
    min_dis_shadowr = min(discounted_shadowrs)
    max_dis_shadowr = max(discounted_shadowrs)

    for t in range(len(r)):
        ax, ay, action = agents_locs[t]
        textloc, triloc = get_location(ax, ay, action)
        # shadow r
        axs[0, 1].text(textloc[0], textloc[1], round(shadow_r[t], 2))
        tri = plt.Polygon(triloc, color=get_color(shadow_r[t]),
                          alpha=get_alpha(shadow_r[t], min_shadow_r, max_shadow_r))
        axs[0, 1].add_patch(tri)
        # total return
        axs[1, 0].text(textloc[0], textloc[1], round(discounted_rewards[t], 2))
        tri = plt.Polygon(triloc, color=get_color(discounted_rewards[t]),
                          alpha=get_alpha(discounted_rewards[t], min_dis_rewards, max_dis_rewards))
        axs[1, 0].add_patch(tri)
        # shadow return
        axs[1, 1].text(textloc[0], textloc[1], round(discounted_shadowrs[t], 2))
        tri = plt.Polygon(triloc, color=get_color(discounted_shadowrs[t]),
                          alpha=get_alpha(discounted_shadowrs[t], min_dis_shadowr, max_dis_shadowr))
        axs[1, 1].add_patch(tri)

    plt.savefig(img_dir + '/' + str(episode) + '.jpg')

    discounted_rewards = np.array(discounted_rewards)

    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
    discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
    log_prob = torch.stack(lp)
    policy_gradient = -log_prob * discounted_rewards

    model.zero_grad()
    policy_gradient.sum().backward()
    optimizer.step()

    episode += 1
