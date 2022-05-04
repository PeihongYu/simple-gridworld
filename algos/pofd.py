import torch
import torch.nn as nn
import numpy as np
from algos.ppo import PPO
from algos.model import Discriminator


class POfD(PPO):
    def __init__(self, env, args, expert, batch_size=512, target_steps=2048, repeat_times=4, prior=None):
        super().__init__(env, args, batch_size, target_steps, repeat_times, prior)

        self.share_reward = False
        self.param_share = True
        self.state_dim = int(env.state_space.shape[0] / env.agent_num)
        self.action_num = env.action_space.n

        self.use_expert = True
        self.pweight = 1
        self.pdecay = 1

        self.expert_state_actions = []
        self.load_expert(expert)

        self.bce_loss = nn.BCELoss()

        self.num_d_epochs = 2
        self.discriminators = []
        self.d_optimizers = []
        for aid in range(env.agent_num):
            self.discriminators.append(Discriminator(self.state_dim, self.action_num))
            self.discriminators[aid].to(self.device)
            self.d_optimizers.append(torch.optim.Adam(self.discriminators[aid].parameters(), self.lr))

    def load_status(self, status):
        for aid in range(self.agent_num):
            self.acmodels[aid].load_state_dict(status["model_state"][aid])
            self.optimizers[aid].load_state_dict(status["optimizer_state"][aid])
            self.discriminators[aid].load_state_dict(status["discriminator_state"][aid])
            self.d_optimizers[aid].load_state_dict(status["d_optimizer_state"][aid])

    def load_expert(self, expert):
        for aid in range(self.agent_num):
            expert_states = torch.tensor(expert["states"][aid], dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert["actions"][aid], dtype=torch.int64, device=self.device)
            expert_actions = torch.eye(self.action_num)[expert_actions].to(self.device)
            self.expert_state_actions.append(torch.cat([expert_states, expert_actions], dim=1))

    def normalize_expert(self, rms):
        epsilon = 1e-8
        expert_state_actions = []
        for aid in range(self.agent_num):
            sid = self.state_dim * aid
            cur_mean = rms.mean[sid:sid+self.state_dim]
            cur_var = rms.var[sid:sid+self.state_dim]
            cur_sa = self.expert_state_actions[aid].clone().detach()
            cur_sa[:, :self.state_dim] = (cur_sa[:, :self.state_dim] - cur_mean) / torch.sqrt(cur_var + epsilon)
            expert_state_actions.append(cur_sa)
        return expert_state_actions

    def concatenate_state_actions(self, states, actions):
        state_actions = []
        for aid in range(self.agent_num):
            actions = torch.eye(self.action_num)[actions[:, aid].to(torch.int64)].to(self.device)
            sid = self.state_dim * aid
            state_actions.append(torch.cat([states[:, sid:sid+self.state_dim], actions], dim=1))
        return state_actions

    def update_parameters(self, buffer, tb_writer=None, clip_grad=False, add_noise=False):
        buf_len = buffer.now_len
        buf_state, buf_reward, buf_action, buf_done = buffer.sample_all()

        buf_state_actions = self.concatenate_state_actions(buf_state, buf_action)
        if self.use_state_norm:
            cur_expert_sas = self.normalize_expert(buffer.state_rms)
        else:
            cur_expert_sas = self.expert_state_actions
        for ep in range(self.num_d_epochs):
            for aid in range(self.agent_num):
                expert_prob = self.discriminators[aid](cur_expert_sas[aid])
                agent_prob = self.discriminators[aid](buf_state_actions[aid])
                term1 = self.bce_loss(agent_prob, torch.ones((buf_state_actions[aid].shape[0], 1), device=self.device))
                term2 = self.bce_loss(expert_prob, torch.zeros((cur_expert_sas[aid].shape[0], 1), device=self.device))
                loss = term1 + term2
                if tb_writer:
                    tb_writer.tb_writer.add_scalar("ep_d_loss" + str(aid), loss, tb_writer.ep_num)
                    tb_writer.tb_writer.add_scalar("ep_prob_agent" + str(aid), agent_prob.mean(), tb_writer.ep_num)
                    tb_writer.tb_writer.add_scalar("ep_prob_expert" + str(aid), expert_prob.mean(), tb_writer.ep_num)
                self.d_optimizers[aid].zero_grad()
                loss.backward()
                self.d_optimizers[aid].step()

        with torch.no_grad():
            d_rewards = -torch.log(
                torch.hstack([self.discriminators[aid](buf_state_actions[aid]) for aid in range(self.agent_num)]))
            if tb_writer:
                d_rewards_mean = d_rewards.mean(axis=0)
                for aid in range(self.agent_num):
                    tb_writer.tb_writer.add_scalar("ep_d_rewards" + str(aid), d_rewards_mean[aid], tb_writer.ep_num)
            buf_reward = torch.hstack([buf_reward, d_rewards])
            buf_value = torch.zeros(buf_action.shape, device=self.device)
            buf_logprob = torch.zeros(buf_action.shape, device=self.device)
            for aid in range(self.agent_num):
                dist, value = self.acmodels[aid](buf_state)
                logprob = dist.log_prob(buf_action[:, aid])
                buf_value[:, aid] = value
                buf_logprob[:, aid] = logprob
            if self.use_value_norm:
                buf_value = self.value_normalizer.denormalize(buf_value)
            if self.use_gae:
                buf_r_sum, buf_advantage = self.compute_reward_gae(buf_len, buf_reward, buf_done, buf_value)
            else:
                buf_r_sum, buf_advantage = self.compute_reward_adv(buf_len, buf_reward, buf_done, buf_value)
            if self.use_value_norm:
                self.value_normalizer.update(buf_r_sum)
            del buf_reward, buf_done

        if self.share_reward:
            adv_mean = buf_advantage.mean(dim=1, keepdim=True)
            buf_advantage = adv_mean.repeat(1, self.agent_num)

        for i in range(self.repeat_times):
            length = int(buf_len // self.num_mini_batch * self.num_mini_batch)
            indices = torch.randperm(length, requires_grad=False, device=self.device).reshape(
                [self.num_mini_batch, int(length / self.num_mini_batch)])
            for ind in indices:
                # indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)
                sb_state = buf_state[ind]
                sb_action = buf_action[ind]
                sb_value = buf_value[ind]
                sb_r_sum = buf_r_sum[ind]
                sb_logprob = buf_logprob[ind]
                sb_advantage = buf_advantage[ind]

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

        if self.param_share and self.agent_num > 1:
            state_dict_all = [self.acmodels[aid].critic.state_dict() for aid in range(self.agent_num)]
            avg_sd = state_dict_all[0].copy()
            for key in state_dict_all[0]:
                avg_sd[key] = torch.mean(torch.stack([state_dict_all[aid][key] for aid in range(self.agent_num)]), dim=0)
            for aid in range(self.agent_num):
                self.acmodels[aid].critic.load_state_dict(avg_sd)
