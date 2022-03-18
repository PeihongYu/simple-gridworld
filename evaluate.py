import os
import time
import numpy as np
from envs.gridworld import GridWorldEnv
import utils

# upperLeftSquare_1a
# centerSquare_1a
# centerSquare_2a
# empty_1a
env_name = "centerSquare_2a"
json_file = "./envfiles/" + env_name + ".json"
env = GridWorldEnv(json_file, True)

algorithm = "PPO"

model_dir = 'outputs/centerSquare_2a_PPO'

acmodels, select_action = utils.load_models(model_dir, env)

EPISODES = 2
STEPS = 100
RENDER = True
save_gif = True
if save_gif:
    from array2gif import write_gif
    frames = []

for episode in range(EPISODES):
    done = False
    state = env.reset()
    lp = []
    r = []

    for step in range(STEPS):
        if RENDER:
            img = env.render()
            if save_gif:
                frames.append(np.moveaxis(img.copy(), 2, 0))
        action = select_action(state["vec"].flatten())
        state, r_, done, i_ = env.step(action)

        if done or env.window.closed:
            state = env.reset()
            break

    if env.window.closed:
        break

if save_gif:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), model_dir + ".gif", fps=10)
    print("Done.")
