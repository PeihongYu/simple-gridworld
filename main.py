import argparse
import numpy as np
import torch

from envs.gridworld import GridWorldEnv
from algos.reinforce import REINFORCE
from algos.ppo import PPO
from algos.pofd import POfD
from algos.base import ExpBuffer
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1)
parser.add_argument("--env", default="centerSquare6x6_1a")
parser.add_argument("--dense_reward", default=False, action='store_true')

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
env_name = args.env
env = GridWorldEnv(env_name, dense_reward=args.dense_reward)
state_dim = env.state_space.shape[0]
action_dim = env.action_space.n
agent_num = env.agent_num

# setup priors
use_prior = args.use_prior
pweight = args.pweight
if use_prior:
    env_prefix = env_name.split("_")[0]
    prior_name = "./envfiles/" + env_prefix + "/" + env_prefix + "_prior"
    prior = []
    if "2a" in env_name:
        prior_ids = [0, 2]
    else:
        prior_ids = list(range(agent_num))
    for aid in range(agent_num):
        prior.append(np.load(prior_name + str(prior_ids[aid]) + ".npy"))
else:
    prior = None

algorithm = args.algo
N = args.N

# setup logging directory
model_dir = "outputs_lava_suboptimal/comparison/" + env_name
if args.dense_reward:
    model_dir += "_dense"
model_dir += "_" + algorithm

model_dir += "_ep" + str(args.ppo_epoch)
model_dir += "_nbatch" + str(args.num_mini_batch)

if use_prior:
    model_dir += "_wprior"
    model_dir += "_N" + str(N)

if args.clip_grad:
    print("apply clip grad.")
    model_dir += "_clipGrad"

if args.add_noise:
    print("apply stochastic update.")
    model_dir += "_gradNoise"

if args.use_state_norm:
    print("apply state normalization.")
    model_dir += "_statenorm"

if args.use_value_norm:
    print("apply value normalization.")
    model_dir += "_valuenorm"

if args.use_gae:
    print("use GAE.")
    model_dir += "_useGAE"

model_dir += "_r0"

use_expert = False
# setup algorithms
if algorithm == "REINFORCE":
    batch_size = 1
    max_len = env.max_steps * batch_size
    algo = REINFORCE(env, batch_size=batch_size, prior=prior)
elif algorithm == "PPO":
    max_len = 4096
    algo = PPO(env, args, batch_size=512, target_steps=max_len, repeat_times=4, prior=prior)
    max_len += env.max_steps
elif algorithm == "POfD":
    use_expert = True
    max_len = 4096
    expert = utils.load_expert(env)
    algo = POfD(env, args, expert, batch_size=512, target_steps=max_len, repeat_times=4)
    max_len += env.max_steps
else:
    raise ValueError("Incorrect algorithm name: {}".format(algorithm))

if use_prior or use_expert:
    algo.pweight = args.pweight
    algo.pdecay = args.pdecay

    model_dir += "_pw" + str(algo.pweight)
    model_dir += "_pd" + str(algo.pdecay)

model_dir += "_seed" + str(args.seed)

if args.run >= 0:
    model_dir += "_run" + str(args.run)

buffer = ExpBuffer(max_len, state_dim, agent_num, args)
tb_writer = utils.tb_writer(model_dir, agent_num, use_prior, use_expert)

try:
    status = torch.load(model_dir + "/best_status.pt", map_location=device)
    algo.load_status(status)
    update = status["update"]
    num_frames = status["num_frames"]
    tb_writer.ep_num = status["num_episode"]
    best_return = status["best_return"]
    if use_prior or use_expert:
        algo.pweight = status["pweight"]
except OSError:
    update = 0
    num_frames = 0
    best_return = -999999


while num_frames < target_frames:
    frames = algo.collect_experiences(buffer, tb_writer)
    algo.update_parameters(buffer, tb_writer, args.clip_grad, args.add_noise)
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
        if algorithm == "POfD":
            status["discriminator_state"] = [discrim.state_dict() for discrim in algo.discriminators]
            status["d_optimizer_state"] = [optimizer.state_dict() for optimizer in algo.d_optimizers]
        if use_prior or use_expert:
            status["pweight"] = algo.pweight
        torch.save(status, model_dir + "/last_status.pt")
        # if update % 3 == 0:
        #     torch.save(status, model_dir + "/status_" + str(update) + ".pt")
        if np.all(avg_returns > best_return):
            best_return = avg_returns.copy()
            torch.save(status, model_dir + "/best_status.pt")
