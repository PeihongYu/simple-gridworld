import torch
import os
from envs.gridworld import GridWorldEnv
from algos.model import ActorModel, ACModel
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# upperLeftSquare_1a
# centerSquare_1a
# centerSquare_2a
# empty_1a
env_name = "centerSquare_2a"
json_file = "./envfiles/" + env_name + ".json"
env = GridWorldEnv(json_file, True)
state_dim = env.state_space.shape[0]
action_dim = env.action_space.n
agent_num = env.agent_num

algorithm = "PPO"

model_dir = 'outputs/centerSquare_2a_PPO_wprior'
# model_dir = 'outputs/gridworld-vec-fix-manhattan-reinforce'

acmodels = []
status = torch.load(model_dir + "/best_status.pt", map_location=device)
if algorithm == "REINFORCE":
    for aid in range(agent_num):
        acmodels.append(ActorModel(env.state_space, env.action_space))
elif algorithm == "PPO":
    for aid in range(agent_num):
        acmodels.append(ActorModel(env.state_space, env.action_space))
else:
    raise ValueError("Incorrect algorithm name: {}".format(algorithm))

for aid in range(agent_num):
    acmodels[aid].load_state_dict(status["model_state"][aid])

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
        actions = [0] * agent_num
        for aid in range(agent_num):
            dist, value = acmodels[aid](state["vec"][aid])
            action = dist.sample()
            actions[aid] = action
        state, r_, done, i_ = env.step(actions)

        if done:
            state = env.reset()
            break

