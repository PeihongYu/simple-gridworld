import numpy as np
from envs.gridworld import GridWorldEnv
import utils

if __name__ == '__main__':
    for aid in range(4):
        env_name = "centerSquare6x6_1a_" + str(aid)
        env = GridWorldEnv(env_name)

        states = np.genfromtxt("trajs/centerSquare6x6_suboptimal_states{0}.csv".format(aid), dtype=np.int32)
        actions = np.genfromtxt("trajs/centerSquare6x6_suboptimal_actions{0}.csv".format(aid), dtype=np.int32)

        lambdas = np.zeros([4, env.width, env.height])

        for i in range(len(states)):
            x, y = states[i]
            a = actions[i]
            lambdas[a, x, y] += 1

        lambdas /= lambdas.sum()

        print(lambdas.sum())

        np.save("priors/centerSquare6x6_suboptimal_prior" + str(aid) + ".npy", lambdas)
