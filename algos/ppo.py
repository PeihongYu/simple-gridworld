import torch
import numpy as np
from algos.base import AgentBase
from algos.model import ACModel


class PPO(AgentBase):
    def __init__(self, env, batch_size=512, target_steps=2048, repeat_times=4, prior=None):
        super().__init__(env, prior)
        self.share_reward = True
        self.batch_size = batch_size        # how many frames for each update
        self.repeat_times = repeat_times    # how many times to reuse the memory
        self.target_steps = target_steps

        self.clip_eps = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.value_loss_coef = 0.5
        self.max_grad_norm = 1

        for aid in range(env.agent_num):
            self.acmodels.append(ACModel(env.state_space, env.action_space))
            self.acmodels[aid].to(self.device)
            self.optimizers.append(torch.optim.Adam(self.acmodels[aid].parameters(), self.lr))

    def select_action(self, state):
        actions = [0] * self.agent_num
        for aid in range(self.agent_num):
            dist, value = self.acmodels[aid](state.flatten())
            action = dist.sample()
            actions[aid] = action
        return actions

    def collect_experiences(self, buffer, tb_writer=None, N=100):
        if self.use_prior:
            self.compute_lambda(N=N)

        buffer.empty_buffer_before_explore()
        steps = 0
        ep_returns = np.zeros(self.agent_num * (1 + self.use_prior))
        while steps < self.target_steps:
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
        return steps

    def update_parameters(self, buffer, tb_writer=None, clip_grad=False, add_noise=False):
        buf_len = buffer.now_len
        with torch.no_grad():
            buf_state, buf_reward, buf_action, buf_done = buffer.sample_all()
            buf_value = torch.zeros(buf_action.shape, device=self.device)
            buf_logprob = torch.zeros(buf_action.shape, device=self.device)
            for aid in range(self.agent_num):
                dist, value = self.acmodels[aid](buf_state)
                logprob = dist.log_prob(buf_action[:, aid])
                buf_value[:, aid] = value
                buf_logprob[:, aid] = logprob
            buf_r_sum, buf_advantage = self.compute_reward_adv(buf_len, buf_reward, buf_done, buf_value)
            del buf_reward, buf_done

        if self.share_reward:
            adv_mean = buf_advantage.mean(dim=1, keepdim=True)
            buf_advantage = adv_mean.repeat(1, self.agent_num)

        for i in range(int(self.repeat_times * buf_len / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)
            sb_state = buf_state[indices]
            sb_action = buf_action[indices]
            sb_value = buf_value[indices]
            sb_r_sum = buf_r_sum[indices]
            sb_logprob = buf_logprob[indices]
            sb_advantage = buf_advantage[indices]

            for aid in range(self.agent_num):
                dist, value = self.acmodels[aid](sb_state)
                entropy = dist.entropy().mean()

                ratio = torch.exp(dist.log_prob(sb_action[:, aid]) - sb_logprob[:, aid])
                surr1 = sb_advantage[:, aid] * ratio
                surr2 = sb_advantage[:, aid] * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = value + torch.clamp(value - sb_value[:, aid], -self.clip_eps, self.clip_eps)
                surr1 = (value - sb_r_sum[:, aid]).pow(2)
                surr2 = (value_clipped - sb_r_sum[:, aid]).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                loss = policy_loss - self.lambda_entropy * entropy + self.value_loss_coef * value_loss
                self.optimizers[aid].zero_grad()
                loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodels[aid].parameters()) ** 0.5
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.acmodels[aid].parameters(), self.max_grad_norm)
                # grad_norm2 = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodels[aid].parameters()) ** 0.5
                # print("agent ", aid, ": grad_norm before and after clipping ...", grad_norm, grad_norm2)
                if add_noise and grad_norm < 1:
                    for params in self.acmodels[aid].parameters():
                        params.grad += torch.randn(params.grad.shape, device=self.device)
                self.optimizers[aid].step()
                if tb_writer:
                    tb_writer.add_grad_info(aid, policy_loss.item(), value_loss.item(), grad_norm)

    def compute_reward_adv(self, buf_len, buf_reward, buf_done, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in reversed(range(buf_len)):
            buf_r_sum[i] = buf_reward[i] + self.gamma * (1 - buf_done[i]) * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        if self.use_prior or self.use_expert:
            buf_r_sum = (1 - self.pweight) * buf_r_sum[:, :self.agent_num] + self.pweight * buf_r_sum[:, self.agent_num:]
            self.pweight *= self.pdecay
        buf_advantage = buf_r_sum - ((1 - buf_done) * buf_value)
        buf_advantage = (buf_advantage - buf_advantage.mean(dim=0)) / (buf_advantage.std(dim=0) + 1e-5)
        return buf_r_sum, buf_advantage

    def compute_reward_gae(self, buf_len, buf_reward, buf_done, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value
        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in reversed(range(buf_len)):
            buf_r_sum[i] = buf_reward[i] + self.gamma * (1 - buf_done[i]) * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + self.gamma * (1 - buf_done[i]) * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean(dim=0)) / (buf_advantage.std(dim=0) + 1e-5)
        return buf_r_sum, buf_advantage
