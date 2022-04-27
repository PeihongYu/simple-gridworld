import os
import time
import random
import numpy as np
from envs.gridworld import GridWorldEnv
import utils


def save_to_file(data, file_path):
    try:
        with open(file_path, "ab") as handle:
            np.savetxt(handle, (data,), fmt="%s")
    except FileNotFoundError:
        with open(file_path, "wb") as handle:
            np.savetxt(handle, (data,), fmt="%s")


def flip_action_random(action, aid):
    action = action[0]
    if aid == 0:
        action = 1 if action == 2 and random.random() < 0.5 else 2
    elif aid == 1:
        action = 0 if action == 2 and random.random() < 0.5 else 2
    elif aid == 2:
        action = 0 if action == 3 and random.random() < 0.5 else 3
    elif aid == 3:
        action = 1 if action == 3 and random.random() < 0.5 else 3
    return [action]


def flip_action(action, aid):
    action = action[0]
    if aid == 0:
        action = 1 if action == 2 else 2
    elif aid == 1:
        action = 0 if action == 2 else 2
    elif aid == 2:
        action = 0 if action == 3 else 3
    elif aid == 3:
        action = 1 if action == 3 else 3
    return [action]


def flip_state(state, aid):
    if aid == 0 or aid == 2:
        new_state = np.array([state[1], state[0]])
    else:
        new_state = np.array([9 - state[1], 9 - state[0]])
    return new_state


aid = 3
env_name = "centerSquare6x6_1a_" + str(aid)
env = GridWorldEnv(env_name, visualize=True)

algorithm = "PPO"

# model_dir = 'outputs/centerSquare6x6_3a_PPO_wprior0.995_N1000_gradNoise'
# model_dir = 'outputs_lava/centerSquare_1a_PPO'
model_dir = 'outputs_lava_suboptimal/centerSquare6x6_1a_' + str(aid) + '_dense_PPO_r0'

acmodels, select_action = utils.load_models(model_dir, env, model=15)

EPISODES = 20
STEPS = 100
RENDER = True
save_gif = False
save_traj = True
rand_threshold = 0.5 if save_traj else 0

if save_gif:
    from array2gif import write_gif
    frames = []

returns = [0] * EPISODES
steps = [0] * EPISODES

for episode in range(EPISODES):
    done = False
    state = env.reset()
    agent = state["vec"].flatten()
    lp = []
    r = []
    ret = np.zeros(env.agent_num)
    fliped = False

    for step in range(STEPS):
        if RENDER:
            img = env.render()
            if save_gif:
                frames.append(np.moveaxis(img.copy(), 2, 0))

        s_vec = state["vec"].flatten()
        if fliped:
            s_vec = flip_state(s_vec, aid)
        action = select_action(s_vec)
        action = [action[i].item() for i in range(len(action))]
        if np.array_equal(agent, s_vec):
            if random.random() < rand_threshold:
                fliped = True
                print("flipped!")
                action = flip_action(action, aid)
        elif fliped:
            action = flip_action(action, aid)

        if save_traj:
            save_to_file(state["vec"].flatten(), "trajs/centerSquare6x6_suboptimal_states" + str(aid) + ".csv")
            save_to_file(action, "trajs/centerSquare6x6_suboptimal_actions" + str(aid) + ".csv")

        state, r_, done, i_ = env.step(action)
        ret += r_

        # print("step ", step, ": ", state["vec"].flatten(), r_, i_)

        if done or env.window.closed:
            state = env.reset()
            break

    returns[episode] = ret
    steps[episode] = step + 1
    print("episode ", episode, ": steps", step+1, ", return", ret)

    if env.window.closed:
        break

print("Averaged episode return over ", EPISODES, " episodes: ", sum(returns)/len(returns))
print("Total number of samples: ", sum(steps))

if save_gif:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), model_dir + ".gif", fps=10)
    print("Done.")
