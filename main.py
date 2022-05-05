import argparse
import numpy as np
import torch

from envs.gridworld import GridWorldEnv
from envs.mpe.environment import MPEEnv
from algos.reinforce import REINFORCE
from algos.ppo import PPO
from algos.pofd import POfD
from algos.base import ExpBuffer
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1)
parser.add_argument("--env", default="centerSquare6x6_1a")
parser.add_argument("--scenario_name", default="simple_spread")
parser.add_argument('--num_agents', default=3, type=int, help='Number of agents')
parser.add_argument("--use_done_func", default=False, action='store_true')
parser.add_argument('--max_steps', default=25, type=int)

parser.add_argument("--dense_reward", default=False, action='store_true')
parser.add_argument("--local_obs", default=False, action='store_true')

parser.add_argument("--algo", default="PPO",
                    help="algorithm to use: POfD | PPO | REINFORCE")
parser.add_argument("--use_prior", default=False, action='store_true')
parser.add_argument("--ppo_epoch", default=4, type=int)
parser.add_argument("--num_mini_batch", default=4, type=int)
parser.add_argument("--pweight", default=0.02, type=float)
parser.add_argument("--pdecay", default=1, type=float)
parser.add_argument("--N", default=1000, type=int)
parser.add_argument("--clip_grad", default=False, action='store_true')
parser.add_argument("--add_noise", default=False, action='store_true')
parser.add_argument("--use_state_norm", default=False, action='store_true')
parser.add_argument("--use_value_norm", default=False, action='store_true')
parser.add_argument("--use_gae", default=False, action='store_true')

parser.add_argument("--frames", type=int, default=4000000)
parser.add_argument('--run', type=int, default=-1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_frames = args.frames

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# create environment
if "centerSquare" in args.env:
    env_name = args.env
    env = GridWorldEnv(env_name, dense_reward=args.dense_reward)
    args.local_obs = False
elif "mpe" in args.env:
    env_name = "mpe_" + args.scenario_name + "_" + str(args.num_agents) + "a"
    env = MPEEnv(args)
    args.local_obs = True
else:
    raise ValueError("Invalid environment name: {}".format(args.env))

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
agent_num = env.agent_num

# setup logging directory
root_dir = "outputs_lava_suboptimal"
model_dir = utils.get_model_dir_name(root_dir, env_name, args)
print("Model save at: ", model_dir)

# setup priors
if args.use_prior:
    prior = utils.load_prior(env_name, use_suboptimal=True)
else:
    prior = None

use_expert_traj = False
# setup algorithms
if args.algo == "REINFORCE":
    batch_size = 1
    max_len = env.max_steps * batch_size
    algo = REINFORCE(env, args, batch_size=batch_size, prior=prior)
elif args.algo == "PPO":
    max_len = 4096
    algo = PPO(env, args, target_steps=max_len, prior=prior)
    max_len += env.max_steps
elif args.algo == "POfD":
    use_expert_traj = True
    max_len = 4096
    expert_traj = utils.load_expert_trajectory(env_name, use_suboptimal=True)
    algo = POfD(env, args, expert_traj, target_steps=max_len)
    max_len += env.max_steps
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

buffer = ExpBuffer(max_len, state_dim, agent_num, args)
tb_writer = utils.tb_writer(model_dir, agent_num, args.use_prior)

# try to load existing models
try:
    status = torch.load(model_dir + "/best_status.pt", map_location=device)
    algo.load_status(status)
    update = status["update"]
    num_frames = status["num_frames"]
    tb_writer.ep_num = status["num_episode"]
    best_return = status["best_return"]
    if args.use_prior or use_expert_traj:
        algo.pweight = status["pweight"]
except OSError:
    update = 0
    num_frames = 0
    best_return = -999999

# start to train
while num_frames < target_frames:
    frames = algo.collect_experiences(buffer, tb_writer)
    algo.update_parameters(buffer, tb_writer)
    num_frames += frames
    avg_returns = tb_writer.log(num_frames)

    update += 1
    if update % 10 == 0:
        tb_writer.log_csv()
        tb_writer.empty_buffer()
        status = {"num_frames": num_frames, "update": update,
                  "num_episode": tb_writer.ep_num, "best_return": best_return,
                  "model_state": [acmodel.state_dict() for acmodel in algo.acmodels],
                  "optimizer_state": [optimizer.state_dict() for optimizer in algo.optimizers]}
        if args.algo == "POfD":
            status["discriminator_state"] = [discrim.state_dict() for discrim in algo.discriminators]
            status["d_optimizer_state"] = [optimizer.state_dict() for optimizer in algo.d_optimizers]
        if args.use_prior or use_expert_traj:
            status["pweight"] = algo.pweight
        torch.save(status, model_dir + "/last_status.pt")
        if update % 10 == 0:
            torch.save(status, model_dir + "/status_" + str(update) + "_" + str(num_frames) + ".pt")
        if np.all(avg_returns > best_return):
            best_return = avg_returns.copy()
            torch.save(status, model_dir + "/best_status.pt")
