import pandas as pd
import numpy as np
import torch
import os
import tensorboardX
from envs.gridworld_2a import GridWorld2a
from algos.model import actorModel

np.set_printoptions(precision=2)

os.makedirs("outputs", exist_ok=True)

DEVICE = "cuda:0"
EPISODES = 10000
STEPS = 1000
GAMMA = 0.99
RENDER = False
batch_size = 100

env = GridWorld2a()
models = []
optimizers = []
for i in range(len(env.agent_pos)):
    models.append(actorModel(4, 4).to(DEVICE))
    optimizers.append(torch.optim.Adam(models[i].parameters(), lr=0.0001))

# model_dir = 'outputs/gridworld-vec-fix-reinforce-wprior'
model_dir = 'outputs/block_2a/gridworld-vec-fix-reinforce-bs' + str(batch_size)
os.makedirs(model_dir, exist_ok=True)

try:
    status = torch.load(model_dir + '/last_params_cloud.pt', map_location=DEVICE)
    # local model and optimizer
    for i in range(2):
        models[i].load_state_dict(status["model_state"][i])
        optimizers[i].load_state_dict(status["optimizer_state"][i])
    # local logging information
    episode = status["episode"]
    all_rewards = status["all_rewards"]
    all_eplens = status["all_eplens"]
    best_rolling = status["best_rolling"]

except OSError:
    episode = 0
    all_rewards = [[], []]
    all_eplens = []
    best_rolling = -99999

bs = 0
lp = [[], []]
r = [[], []]
dones = []
tb_writer = tensorboardX.SummaryWriter(model_dir)

while episode < EPISODES:

    done = False
    step = 0
    state = env.reset()

    while not done:
        if RENDER:
            env.render()
        action_2a = []
        for i in range(2):
            actions = models[i](state["vec"])
            action, log_prob = models[i].get_action(actions)
            action_2a.append(action)
            lp[i].append(log_prob)

        state, r_, done, i_ = env.step(action_2a)
        for i in range(2):
            r[i].append(r_[i])
        dones.append(done)
        step += 1

        if done:
            bs += 1
            state = env.reset()
            for i in range(2):
                all_rewards[i].append(np.sum(r[i][-step:]))
            all_eplens.append(step)
            step = 0
            print(f"EPISODE {episode} STEP: {pd.Series(all_eplens).tail(100).mean():.2f} SCORE: {all_rewards[0][-1]:.2f}, {all_rewards[1][-1]:.2f} "
                  f"roll: {pd.Series(all_rewards[0]).tail(100).mean():.2f}, {pd.Series(all_rewards[1]).tail(100).mean():.2f}")
            tb_writer.add_scalar("frames", pd.Series(all_eplens).tail(100).mean(), episode)
            tb_writer.add_scalar("return_a1", pd.Series(all_rewards[0]).tail(100).mean(), episode)
            tb_writer.add_scalar("return_a2", pd.Series(all_rewards[1]).tail(100).mean(), episode)

            if episode % 100 == 0:
                status = {"episode": episode,
                          "model_state": [models[0].state_dict(), models[1].state_dict()],
                          "optimizer_state": [optimizers[0].state_dict(), optimizers[0].state_dict()],
                          "all_rewards": all_rewards,
                          "all_eplens": all_eplens,
                          "best_rolling": best_rolling}
                torch.save(status, model_dir + '/last_params_cloud.pt')
                cur_avg = (pd.Series(all_rewards[0]).tail(100).mean() + pd.Series(all_rewards[1]).tail(100).mean())
                if cur_avg > best_rolling:
                    best_rolling = cur_avg
                    print("saving...")
                    torch.save(status, model_dir + '/best_params_cloud.pt')

            break

    if bs == batch_size:

        if not (np.min(r) == 0 and np.max(r) == 0):
            discounted_returns = np.array([0.0] * len(r[0]))

            for i in range(2):
                cur_discounted_returns = [0] * len(r[i])
                for t in reversed(range(len(r[i]))):
                    cur_discounted_returns[t] = r[i][t] + GAMMA * cur_discounted_returns[t + 1] if not dones[t] else r[i][t]
                discounted_returns += cur_discounted_returns
            discounted_returns /= 2

            discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32, device=DEVICE)
            discounted_returns = (discounted_returns - torch.mean(discounted_returns)) / (torch.std(discounted_returns))
            for i in range(2):
                log_prob = torch.stack(lp[i])
                policy_gradient = -log_prob * discounted_returns

                models[i].zero_grad()
                policy_gradient.sum().backward()
                optimizers[i].step()

        bs = 0
        lp = [[], []]
        r = [[], []]
        dones = []




    episode += 1