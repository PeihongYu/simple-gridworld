import torch
import matplotlib.pyplot as plt
from algos.model import ActorModel, ACModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(model_dir, env):
    acmodels = []
    status = torch.load(model_dir + "/best_status.pt", map_location=device)
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

    elif "PPO" in model_dir:
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


def draw_policy(agent, goal, model, ax):
    # agent
    self_y, self_x = agent
    ax.fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')

    # goal
    goal_y, goal_x = goal
    ax.fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')

    goal = env.goal

