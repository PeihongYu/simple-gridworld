import torch
import numpy as np

from algos.base import AgentBase, ExpBuffer
from algos.model import ActorModel


class REINFORCE(AgentBase):
    def __init__(self, env, batch_size=1, prior=None):
        super().__init__(env, prior)
        self.share_reward = True
        self.batch_size = batch_size    # how many episodes for each update

        for aid in range(env.agent_num):
            self.acmodels.append(ActorModel(env.state_space, env.action_space))
            self.acmodels[aid].to(self.device)
            self.optimizers.append(torch.optim.Adam(self.acmodels[aid].parameters(), self.lr))

    def select_action(self, state):
        actions = [0] * self.agent_num
        for aid in range(self.agent_num):
            dist = self.acmodels[aid](state.flatten())
            action = dist.sample()
            actions[aid] = action
        return actions

    def collect_experiences(self, buffer, tb_writer=None, N=100):
        if self.use_prior:
            self.compute_lambda(N=N)

        buffer.empty_buffer_before_explore()
        episodes = 0
        steps = 0
        ep_returns = np.zeros(self.agent_num * (1 + self.use_prior))
        while episodes < self.batch_size:
            state = self.env.reset()
            done = False
            ep_steps = 0
            ep_returns *= 0
            while not done:
                action = self.select_action(state["vec"])
                next_state, reward, done, _ = self.env.step(action)
                if self.use_prior:
                    shadow_reward = self.compute_shadow_r(state["vec"], action)
                    reward = reward + shadow_reward
                ep_returns += reward
                buffer.append(state["vec"], action, reward, done)
                state = next_state
                steps += 1
                ep_steps += 1
            if tb_writer:
                tb_writer.add_info(ep_steps, ep_returns, self.pweight)
            episodes += 1
        return steps

    def update_parameters(self, buffer, tb_writer=None, clip_grad=False, add_noise=False):
        buf_len = buffer.now_len
        with torch.no_grad():
            buf_state, buf_reward, buf_action, buf_done = buffer.sample_all()
            buf_r_sum, normalized_r_sum = self.compute_reward_sum(buf_len, buf_reward, buf_done)

        if self.share_reward:
            r_sum_mean = normalized_r_sum.mean(dim=1, keepdim=True)
            normalized_r_sum = r_sum_mean.repeat(1, self.agent_num)

        for aid in range(self.agent_num):
            dist = self.acmodels[aid](buf_state)
            log_prob = dist.log_prob(buf_action[:, aid])
            policy_gradient = -log_prob * normalized_r_sum[:, aid]

            # print(self.acmodels[aid].parameters())
            # for name, param in self.acmodels[aid].named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            self.optimizers[aid].zero_grad()
            policy_gradient.sum().backward()
            self.optimizers[aid].step()

    def compute_reward_sum(self, buf_len, buf_reward, buf_done) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)
        pre_r_sum = 0  # reward sum of previous step
        for i in reversed(range(buf_len)):
            buf_r_sum[i] = buf_reward[i] + self.gamma * (1 - buf_done[i]) * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            # buf_r_sum[i] = buf_reward[i] + self.gamma * buf_r_sum[i + 1] if not buf_done[i] else buf_reward[i]

        normalized_r_sum = (buf_r_sum - buf_r_sum.mean(dim=0)) / (buf_r_sum.std(dim=0) + 1e-5)
        return buf_r_sum, normalized_r_sum
