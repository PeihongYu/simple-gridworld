import numpy as np
import torch

from algos.occupancy_measure import StateOccupancyMeasure


class ExpBuffer:
    def __init__(self, max_len, state_dim, agent_num, use_prior=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.state = torch.empty((max_len, state_dim * agent_num), dtype=torch.float32, device=self.device)
        self.action = torch.empty((max_len, agent_num), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((max_len, agent_num + agent_num * use_prior), dtype=torch.float32, device=self.device)
        self.done = torch.empty((max_len, 1), dtype=torch.float32, device=self.device)

    def append(self, state, action, reward, done):
        if self.now_len >= self.max_len:
            return
        self.state[self.now_len] = torch.as_tensor(state, device=self.device).flatten()
        self.action[self.now_len] = torch.as_tensor(action, device=self.device).flatten()
        self.reward[self.now_len] = torch.as_tensor(reward, device=self.device).flatten()
        self.done[self.now_len] = torch.as_tensor(done, device=self.device)

        self.now_len += 1

    def sample_batch(self, batch_size):
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device)
        return self.state[indices], self.reward[indices], self.action[indices], self.done[indices]

    def sample_all(self):
        return self.state[:self.now_len], self.reward[:self.now_len], \
               self.action[:self.now_len], self.done[:self.now_len]

    def empty_buffer_before_explore(self):
        self.now_len = 0


class AgentBase:
    def __init__(self, env, prior=None):
        self.env = env
        self.agent_num = env.agent_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 0.0001  # the learning rate for optimizers
        self.gamma = 1  #0.99  # the discount factor for future rewards
        self.lambda_gae_adv = 0.98  # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)

        self.acmodels = []
        self.optimizers = []

        if prior is None:
            self.use_prior = False
        else:
            self.use_prior = True
            self.prior = prior
            self.pweight = 0.995
            self.pdecay = 0.995

        if self.use_prior:
            self.occupancy_measures = StateOccupancyMeasure(env.grid.shape, env.agent_num)

    def load_status(self, status):
        for aid in range(self.agent_num):
            self.acmodels[aid].load_state_dict(status["model_state"][aid])
            self.optimizers[aid].load_state_dict(status["optimizer_state"][aid])

    def select_action(self, state):
        pass

    def get_prior_prob(self, state, action):
        prob = [0] * self.agent_num
        for aid in range(self.agent_num):
            prob[aid] = self.prior[aid][action[aid], state[aid][0], state[aid][1]]
        return prob

    def compute_shadow_r(self, state, action):
        shadow_r = [0] * self.agent_num
        cur_probs = self.occupancy_measures.get_prob(state["vec"])
        prior_probs = self.get_prior_prob(state["vec"], action)
        for aid in range(self.agent_num):
            cur_prob = cur_probs[aid] + 1e-12
            prior_prob = prior_probs[aid] + 1e-12
            if cur_prob != prior_prob:
                shadow_r[aid] = - (np.log(2 * cur_prob) - np.log(cur_prob + prior_prob))
        return shadow_r

    def compute_lambda(self, N=100):
        episode = 0
        while episode < N:
            state = self.env.reset()
            self.occupancy_measures.count_cur_state(state["vec"])
            done = False
            while not done:
                action = self.select_action(state["vec"])
                state, reward, done, _ = self.env.step(action)
                self.occupancy_measures.count_cur_state(state["vec"])
            self.occupancy_measures.update_lambdas()
            episode += 1
        self.occupancy_measures.normalize()
