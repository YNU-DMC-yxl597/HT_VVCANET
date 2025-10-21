import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import  make_env

""" Fitted Policy Iteration for Combinatorial Menus """
class FPI:
    def __init__(self, agent,old_agent, rollout_buffer, args, v_dist,env_class,mask_N):

        self.agent = agent
        self.old_agent = old_agent
        self.rollout_buffer = rollout_buffer
        self.args = args
        
        self.device = args.device
        self.env_steps = args.env_steps
        self.v_dist = v_dist

        self.N = args.num_users
        self.num_VMs = args.num_VMs
        self.num_PMs = args.num_PMs
        self.num_resources = args.num_resources
        self.T = args.num_times


        self.device=args.device
        self.eps_clip=0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.mse_loss = nn.MSELoss()

        self.init_optimizers(self.args.lr_vf, self.args.lr_pi)

        self.eval_envs = [make_env(env_class, args, v_dist, i, mask_N,False) for i in range(args.test_num_envs)]
        mask = torch.ones(self.N + 1,4+self.num_VMs*self.T, device=args.device)
        mask[mask_N, :] = 0
        self.mask_N = mask_N
        self.mask = mask[:-1, :]

    def collect_rollouts(self,envs):

        self.rollout_buffer.reset()
        obs, _ = envs.reset()
        obs = torch.Tensor(obs).to(self.device)
        done = torch.zeros(self.env_steps).to(self.device)

        for _ in range(self.T):

            with torch.no_grad():
                obs[:, :self.N * (4+self.num_VMs*self.T)] = obs[:, :self.N * (4+self.num_VMs*self.T)]*(self.mask.reshape(-1).unsqueeze(0))
                action,log_probs,value = self.old_agent.get_action_and_value(obs)
                value = value.flatten()

            next_obs, reward, terminations, truncations,info = envs.step(action)


            reward = torch.tensor(reward).to(self.device).view(-1)
            self.rollout_buffer.add(obs,action, reward, done, value,log_probs)

            done = np.logical_or(terminations, truncations)
            obs, done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

        with torch.no_grad():
            next_value = torch.zeros(self.env_steps).to(self.device)
            self.rollout_buffer.compute_returns_and_advantages(next_value, done)


    def init_optimizers(self,  lr_pi):
        self.opt_pi = torch.optim.Adam(self.agent.parameters(), lr = lr_pi)


    def fit_policy(self, pi_epochs):

        batch_size = self.args.batch_size

        for epoch in range(pi_epochs):
            for mb_obs,mb_actions, mb_advantages,mb_returns, _,mb_old_log_probs in self.rollout_buffer.get(batch_size = batch_size):

                hidden = self.agent(mb_obs)
                logits = self.agent.fc_actor(hidden)

                action_x_ij = logits.reshape(-1, self.N, 2)
                action_x_ij = F.softmax(action_x_ij, dim=-1)

                dist = torch.distributions.Categorical(probs=action_x_ij)
                log_probs = dist.log_prob(mb_actions[:,:self.N]).sum(-1)
                entropy = dist.entropy().mean()


                values = self.agent.get_value(mb_obs).squeeze(-1)


                ratios = torch.exp(log_probs - mb_old_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                    1 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(values, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt_pi.zero_grad()
                loss.backward()
                self.opt_pi.step()
        self.old_agent.load_state_dict(self.agent.state_dict())
            
    def learn(self,envs):

        self.collect_rollouts(envs)
        self.fit_policy(self.args.pi_epochs)


