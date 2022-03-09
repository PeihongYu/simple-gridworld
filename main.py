import numpy as np
import torch

from envs.gridworld import GridWorldEnv
from algos.reinforce import REINFORCE
from algos.ppo import PPO
from algos.base import ExpBuffer
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_frames = 4000000

# upperLeftSquare_1a
# centerSquare_1a
# centerSquare_2a
# empty_1a
env_name = "centerSquare_4a"
json_file = "./envfiles/" + env_name + ".json"
env = GridWorldEnv(json_file)
state_dim = env.state_space.shape[0]
action_dim = env.action_space.n
agent_num = env.agent_num

use_prior = True
pweigt = 0.998
if use_prior:
    # prior_names = ["centerSquare_1a", "centerSquare_1a_flip"]
    prior = []
    for aid in range(agent_num):
        prior.append(np.load("./envfiles/" + env_name + "_prior" + str(aid) + ".npy"))
else:
    prior = None

algorithm = "PPO"

model_dir = "outputs/" + env_name + "_" + algorithm
if use_prior:
    model_dir += "_wprior" + str(pweigt)

if algorithm == "REINFORCE":
    batch_size = 1
    max_len = env.max_steps * batch_size
    algo = REINFORCE(env, batch_size=batch_size, prior=prior)
elif algorithm == "PPO":
    max_len = 4096
    algo = PPO(env, batch_size=512, target_steps=max_len, repeat_times=4, prior=prior)
    max_len += env.max_steps
else:
    raise ValueError("Incorrect algorithm name: {}".format(algorithm))

if use_prior:
    algo.pdecay = pweigt

buffer = ExpBuffer(max_len, state_dim, agent_num, use_prior)
tb_writer = utils.tb_writer(model_dir, agent_num, use_prior)


try:
    status = torch.load(model_dir + "/best_status.pt", map_location=device)
    algo.load_status(status)
    update = status["update"]
    num_frames = status["num_frames"]
    tb_writer.ep_num = status["num_episode"]
    best_return = status["best_return"]
    if use_prior:
        algo.pweight = status["pweight"]
except OSError:
    update = 0
    num_frames = 0
    best_return = -999999


while num_frames < target_frames:
    frames = algo.collect_experiences(buffer, tb_writer)
    algo.update_parameters(buffer)
    num_frames += frames
    avg_returns = tb_writer.log(num_frames)

    if update % 10 == 0:
        status = {"num_frames": num_frames, "update": update,
                  "num_episode": tb_writer.ep_num, "best_return": best_return,
                  "model_state": [acmodel.state_dict() for acmodel in algo.acmodels],
                  "optimizer_state": [optimizer.state_dict() for optimizer in algo.optimizers]}
        if use_prior:
            status["pweight"] = algo.pweight
        torch.save(status, model_dir + "/last_status.pt")
        if np.all(avg_returns > best_return):
            best_return = avg_returns.copy()
            torch.save(status, model_dir + "/best_status.pt")


