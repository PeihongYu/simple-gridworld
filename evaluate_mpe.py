import os
import time
import numpy as np
import argparse
from envs.gridworld import GridWorldEnv
from envs.mpe.environment import MPEEnv
import utils


def save_to_file(data, file_path):
    try:
        with open(file_path, "ab") as handle:
            np.savetxt(handle, (data,), fmt="%s")
    except FileNotFoundError:
        with open(file_path, "wb") as handle:
            np.savetxt(handle, (data,), fmt="%s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--scenario_name', default='simple_spread')
    parser.add_argument('--num_agents', default=3, type=int, help='Number of agents.')
    parser.add_argument("--use_done_func", default=False, action='store_true')
    parser.add_argument('--max_steps', default=100, type=int)
    args = parser.parse_args()

    env_name = "mpe_" + args.scenario_name + "_3a"
    env = MPEEnv(args)

    algorithm = "PPO"

    model_dir = 'outputs_lava_suboptimal/mpe_simple_spread_3a_PPO_ep4_nbatch4_seed1'

    model_name = "best"
    acmodels, select_action = utils.load_models(model_dir, env, model=model_name, use_local_obs=True)

    EPISODES = 40
    STEPS = 100
    RENDER = True
    save_gif = False
    save_traj = False

    if save_gif:
        from array2gif import write_gif
        frames = []

    returns = [0] * EPISODES
    steps = [0] * EPISODES

    if RENDER:
        env.render()

    for episode in range(EPISODES):
        done = False
        state = env.reset()
        lp = []
        r = []
        ret = np.zeros(env.agent_num)

        for step in range(STEPS):
            if RENDER:
                env.render()
                # time.sleep(0.2)
            action = select_action(state["vec"])

            if save_traj:
                if model_name != "best":
                    model_name = "suboptimal"
                save_to_file(state["vec"].flatten(), "trajs/" + env_name + "_" + model_name + "_states.csv")
                save_to_file([action[i].item() for i in range(len(action))],
                             "trajs/" + env_name + "_" + model_name + "_actions.csv")

            state, r_, done, i_ = env.step(action)
            ret += r_

            # print("step ", step, ": ", state["vec"].flatten(), r_, i_)

            if done:
                state = env.reset()
                break

        returns[episode] = ret
        steps[episode] = step + 1
        print("episode ", episode, ": steps", step+1, ", return", ret)

    print("Averaged episode return over ", EPISODES, " episodes: ", sum(returns)/len(returns))
    print("Total number of samples: ", sum(steps))

    if save_gif:
        print("Saving gif... ", end="")
        write_gif(np.array(frames), model_dir + ".gif", fps=10)
        print("Done.")
