import os
import time
from envs.gridworld import GridWorldEnv
import utils


# upperLeftSquare_1a
# centerSquare_1a
# centerSquare_2a
# empty_1a
env_name = "centerSquare_4a"
json_file = "./envfiles/" + env_name + ".json"
env = GridWorldEnv(json_file, True)

algorithm = "PPO"

model_dir = 'outputs/centerSquare_4a_PPO_wprior0.998'

acmodels, select_action = utils.load_models(model_dir, env)

EPISODES = 1000
STEPS = 100
RENDER = True
for episode in range(EPISODES):
    done = False
    state = env.reset()
    lp = []
    r = []

    for step in range(STEPS):
        if RENDER:
            env.render()
        action = select_action(state["vec"].flatten())
        state, r_, done, i_ = env.step(action)

        if done:
            state = env.reset()
            break

