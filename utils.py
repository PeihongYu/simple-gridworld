import numpy as np
from tqdm import tqdm
import math


def calculate_lambda_parallel(env, model, N=5000):

    lambdas = np.zeros([env.envs[0].height, env.envs[0].width])

    state = env.reset()
    ss = np.array([state[i]["vec"] for i in range(len(env.envs))])
    episode = 0
    while episode < N:
        actions = model(ss)
        actions = [model.get_action(actions[i, :])[0] for i in range(len(ss))]
        # obs, reward, done, info = env.step(actions)
        # for eid in range(len(env.envs)):
        #     if done[eid]:
        #         episode += 1
        #         lambdas += info[eid] / info[eid].sum()
        results = env.step(actions)
        for res in results:
            if res[2]:
                episode += 1
                lambdas += res[3] / res[3].sum()
        ss = np.array([res[0]["vec"] for res in results])

    lambdas /= lambdas.sum()
    lambdas += 1e-9
    lambdas /= lambdas.sum()
    return lambdas


def calculate_lambda(env, model, N=1000):
    lambdas = np.zeros([env.height, env.width])

    for episode in tqdm(range(N)):
        done = False
        state = env.reset()
        steps = 0

        local_lambdas = np.zeros([env.height, env.width])
        i, j = env.agent_pos
        local_lambdas[i, j] += 1

        while not done and steps < 100:
            steps += 1
            actions = model(state["vec"])
            action, log_prob = model.get_action(actions)
            state, r_, done, i_ = env.step(action)

            i, j = env.agent_pos
            local_lambdas[i, j] += 1

            if done:
                break

        lambdas += local_lambdas / local_lambdas.sum()

    lambdas /= lambdas.sum()
    lambdas += 1e-9
    lambdas /= lambdas.sum()
    return lambdas
