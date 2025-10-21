import numpy as np
import torch
import gc
""" Rollout Buffer """
class RolloutBuffer:
    def __init__(self, gamma, gae_lambda,args, device = "cuda"):
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_steps=args.num_times
        self.env_steps = args.env_steps

        self.N = args.num_users
        self.num_VMs = args.num_VMs
        self.num_PMs = args.num_PMs
        self.num_resources = args.num_resources
        self.T = args.num_times

        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.num_steps,self.env_steps,self.N * (4+self.num_VMs*self.T)+ self.num_PMs*self.num_resources+self.num_VMs*self.num_resources + 1).to(self.device)
        self.action= torch.zeros(self.num_steps,self.env_steps,self.N).to(self.device)

        self.rewards = torch.zeros((self.num_steps, self.env_steps,)).to(self.device)
        self.dones = torch.zeros((self.num_steps,self.env_steps,)).to(self.device)
        self.values = torch.zeros((self.num_steps,self.env_steps, )).to(self.device)
        self.advantages = torch.zeros((self.num_steps,self.env_steps, )).to(self.device)
        self.returns = torch.zeros((self.num_steps, self.env_steps,)).to(self.device)
        self.logprobs=torch.zeros((self.num_steps,self.env_steps,)).to(self.device)

        self.pos = 0
        self.full = False
        self.generator_ready = False

    def size(self):
        return self.pos

    def add(self, obs, action,rewards, dones, values,logprobs):
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.logprobs[self.pos] = logprobs
        self.dones[self.pos] = dones
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values

        self.pos += 1
        if self.pos == self.num_steps - 1:
            self.full = True

    def get(self, batch_size = None, return_inds = False):
        assert self.full
        if not self.generator_ready:
            self.obs = self.obs.reshape(-1, self.N * (4+self.num_VMs*self.T)+ self.num_PMs*self.num_resources+self.num_VMs*self.num_resources + 1)
            self.action = self.action.reshape(-1,self.N )
            self.logprobs = self.logprobs.reshape(-1)
            self.advantages = self.advantages.reshape(-1)
            self.returns = self.returns.reshape(-1)
            self.values = self.values.reshape(-1)
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.num_steps*self.env_steps

        start_idx = 0
        batch_num=self.num_steps*self.env_steps // batch_size
        i_th=0
        while i_th< batch_num :
            yield self._get_samples(np.arange(start_idx , start_idx + batch_size), return_inds = return_inds)
            start_idx += batch_size
            i_th+=1


    def _get_samples(self, batch_inds, return_inds):
        if not return_inds:
            return self.obs[batch_inds], self.action[batch_inds],  self.advantages[batch_inds], self.returns[batch_inds], self.values[batch_inds],self.logprobs[batch_inds]
        else:
            return self.obs[batch_inds], self.action[batch_inds], self.advantages[batch_inds], self.returns[batch_inds], self.values[batch_inds],self.logprobs[batch_inds], batch_inds


    def compute_returns_and_advantages(self, last_value, done):
        last_value = last_value.detach()
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values


