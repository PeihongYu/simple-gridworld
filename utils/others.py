import torch
import numpy as np
import matplotlib.pyplot as plt
from algos.model import ActorModel, ACModel, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_expert(env):
    if env.agent_num == 2:
        prior_ids = [0, 2]
    else:
        prior_ids = list(range(env.agent_num))
    expert_states = []
    expert_actions = []
    for id in prior_ids:
        states = np.genfromtxt("trajs/centerSquare6x6_suboptimal_states{0}.csv".format(id))
        actions = np.genfromtxt("trajs/centerSquare6x6_suboptimal_actions{0}.csv".format(id), dtype=np.int32)
        expert_states.append(states)
        expert_actions.append(actions)
    expert = {"states": expert_states, "actions": expert_actions}
    return expert


def load_models(model_dir, env, model="best"):
    acmodels = []
    if model == "best":
        status = torch.load(model_dir + "/best_status.pt", map_location=device)
    elif model == "last":
        status = torch.load(model_dir + "/last_status.pt", map_location=device)
    else:
        status = torch.load(model_dir + "/status_" + str(model) +".pt", map_location=device)
    if "REINFORCE" in model_dir:
        for aid in range(env.agent_num):
            acmodels.append(ActorModel(env.state_space, env.action_space))

        def select_action(state):
            actions = [0] * env.agent_num
            for aid in range(env.agent_num):
                dist = acmodels[aid](state.flatten())
                action = dist.sample()
                actions[aid] = action
            return actions

    elif "PPO" or "POfD" in model_dir:
        for aid in range(env.agent_num):
            acmodels.append(ACModel(env.state_space, env.action_space))

        def select_action(state):
            actions = [0] * env.agent_num
            for aid in range(env.agent_num):
                dist, value = acmodels[aid](state.flatten())
                action = dist.sample()
                actions[aid] = action
            return actions

    else:
        raise ValueError("No such algorithm!")

    for aid in range(env.agent_num):
        acmodels[aid].load_state_dict(status["model_state"][aid])
        acmodels[aid].to(device)

    return acmodels, select_action


def load_discriminator(model_dir, env, model="best"):
    discriminarors = []
    if model == "best":
        status = torch.load(model_dir + "/best_status.pt", map_location=device)
    elif model == "last":
        status = torch.load(model_dir + "/last_status.pt", map_location=device)
    else:
        status = torch.load(model_dir + "/status_" + str(model) +".pt", map_location=device)

    state_dim = int(env.state_space.shape[0] / env.agent_num)
    action_num = env.action_space.n

    for aid in range(env.agent_num):
        discriminarors.append(Discriminator(state_dim, action_num))
        discriminarors[aid].load_state_dict(status["discriminator_state"][aid])
        discriminarors[aid].to(device)

    return discriminarors


# def draw_policy(agent, goal, model, ax):
#     # agent
#     self_y, self_x = agent
#     ax.fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')
#
#     # goal
#     goal_y, goal_x = goal
#     ax.fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')
#
#     goal = env.goal

