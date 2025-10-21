import random
import argparse

import copy
import numpy as np

import torch

import gymnasium as gym

from utils import  make_env
import bundle
from buffer import RolloutBuffer
from net import ActorCriticNetworkBundle
from fpi import FPI
from distribution import VDistBase

""" Hyper-parameters """
class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b

class Args:

    num_users : int =6
    num_VMs : int =2
    num_PMs : int =6
    num_resources : int =3
    num_times: int =5
    batch_size: int = num_users * num_times

    env_steps: int = 16


    log_std_init: float = -2

    num_hidden_units: int = 256

    num_hidden_layers: int = 3

    d_model: int = 4


    lr_vf: float = 1e-3
    lr_pi: float = 1e-4

    vf_epochs: int = 100
    td_epochs = 50
    pi_epochs: int = 5


    train_num_envs: int = 960
    test_num_envs: int = 960

    gamma: float = 1.0

    gae_lambda: float = 0.95

    tau: float = 100

    num_samples_for_pi: int = 256

    log_std_decay: float = 0.25

    max_iteration: int = 10

    lambda_iteration: int = 1

    device: str = "cuda"

    seed: int = 24


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    args = Args()

    v_dist = VDistBase(args,args.device)
    env_class = bundle.AuctionEnv
    policy_class = ActorCriticNetworkBundle
    model_class = FPI

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    agent = [policy_class(args.num_hidden_layers, args.num_hidden_units, args).to(args.device) for i in
             range(args.num_users + 1)]
    agent0 = policy_class(args.num_hidden_layers, args.num_hidden_units, args).to(args.device)
    for ag in agent:
        ag.load_state_dict(agent0.state_dict())
    rollout_buffer = RolloutBuffer(args.gamma, args.gae_lambda, args, args.device)


    old_agent=[policy_class(args.num_hidden_layers, args.num_hidden_units, args).to(args.device) for i in
             range(args.num_users + 1)]
    for i in range(args.num_users + 1):
        old_agent[i].load_state_dict(agent[i].state_dict())
    rollout_buffer = RolloutBuffer(args.gamma, args.gae_lambda, args, args.device)

    for j in range(args.lambda_iteration):

        for k in range(args.num_users,args.num_users + 1):
            model = model_class(agent[k],old_agent[k], rollout_buffer, args, v_dist, env_class, k)
            eval_envs = [make_env(env_class, args, v_dist, i, k, False) for i in range(args.test_num_envs)]

            for iteration in range(args.max_iteration):
                iter_num = args.train_num_envs // args.env_steps
                for i in range(iter_num):
                    envs = gym.vector.SyncVectorEnv(
                        [make_env(env_class, args, v_dist, i * args.env_steps + j, k, True) for j in
                         range(args.env_steps)])
                    model.learn(envs)


        welfare = 0
        paysss = 0
        win_num = 0
        for iteration in range(args.test_num_envs):
            pay = torch.zeros((args.num_users + 1, args.num_times), device=args.device)
            pay_bids = torch.zeros(args.num_users, device=args.device)

            for mask_k in range(args.num_users + 1):
                mask = torch.ones(args.num_users + 1, 4 + args.num_VMs * args.num_times, device=args.device)
                mask[mask_k, :] = 0
                mask_N = mask_k
                mask = mask[:-1, :]
                state = torch.zeros(args.num_users * (4+args.num_VMs*args.num_times)+ args.num_PMs*args.num_resources+args.num_VMs*args.num_resources + 1, device=args.device)
                state[-1] = 1.0
                Request_list,Q,C = v_dist.sample(iteration, False)
                allocated = []
                for j in range(args.num_users):
                    a = Request_list[j].a
                    d = Request_list[j].d
                    e = Request_list[j].e
                    b = Request_list[j].b
                    if state[-1] >= a and state[-1] <= d - e + 1 and j not in allocated and j!=mask_k:
                        S = np.zeros((args.num_VMs, args.num_times))
                        for k in range(args.num_VMs):
                            for t in range(Request_list[j].e):
                                S[k, t] = Request_list[j].S[k][t]
                        time_array = np.array([a, d, e, b])
                        S_time_array = np.hstack((time_array, S.reshape(-1)))
                        state[j * (4 + args.num_VMs * args.num_times):(j + 1) * (
                                    4 + args.num_VMs * args.num_times)] = torch.tensor(S_time_array,device=args.device)

                state[args.num_users * (4 + args.num_VMs * args.num_times):args.num_users * (
                            4 + args.num_VMs * args.num_times) + args.num_PMs * args.num_resources] = torch.tensor(C.reshape(-1),device=args.device)
                state[args.num_users * (
                            4 + args.num_VMs * args.num_times) + args.num_PMs * args.num_resources:-1] = torch.tensor(Q.reshape(-1),device=args.device)

                wait_allocate_resources =(torch.tensor(C,device=args.device).unsqueeze(0)).repeat(args.num_times,1,1)

                while state[-1] <= args.num_times:
                    obs = torch.Tensor(state).to(args.device).unsqueeze(0)
                    with torch.no_grad():
                        obs[:, :args.num_users * (4 + args.num_VMs * args.num_times)] = obs[:,:args.num_users * (4 + args.num_VMs * args.num_times)] * (
                                                                            mask.reshape(-1).unsqueeze(0))
                        action, log_probs, value = agent[mask_k].get_action_and_value(obs)
                    reward = 0
                    for i in range(args.num_users):
                        flag = 0
                        a = Request_list[i].a
                        d = Request_list[i].d
                        e = Request_list[i].e
                        b = Request_list[i].b
                        if action[0][i] == 1 and state[-1] >= a and state[
                            -1] <= d - e + 1 and i not in allocated  and i!=mask_k:
                            C_temp_times = copy.deepcopy(wait_allocate_resources)
                            for j in range(int(state[-1]) - 1,int(state[-1])-1+e):
                                C_temp = copy.deepcopy(C_temp_times[j])
                                for q in range(args.num_VMs):
                                    flag_i = 1
                                    for o in range(Request_list[i].S[q][j - int(state[-1] - 1)]):
                                        flag_i = 0
                                        for p in range(args.num_PMs):
                                            # r_i = 0
                                            flag_i_p = 0
                                            C_temp_i = copy.deepcopy(C_temp)
                                            for r in range(args.num_resources):
                                                # a = Request_list[k].S[q][j - int(X_t[k].item())] * Q[q][r]
                                                r_i = Q[q][r]
                                                if C_temp[p, r] < r_i:
                                                    flag_i_p = 1
                                                    flag_i = 0
                                                    break
                                                else:
                                                    C_temp_i[p, r] -= r_i
                                            if flag_i_p == 0:
                                                flag_i = 1
                                                C_temp = C_temp_i
                                                break
                                        if flag_i == 0:
                                            break
                                    if flag_i == 0:
                                        flag = 1
                                        break
                                if flag == 1:
                                    break
                                else:
                                    C_temp_times[j] = C_temp
                            if flag == 0:
                                wait_allocate_resources = C_temp_times
                                allocated.append(i)
                                reward = reward + b
                                pay[mask_k,int(state[-1]) - 1] += b
                                if mask_k == args.num_users:
                                    pay_bids[i] = b
                                    win_num += 1
                                    welfare+=b

                    state[:args.num_users * (
                            4 + args.num_VMs * args.num_times) + args.num_PMs * args.num_resources] =torch.zeros(
                        args.num_users * (
                                4 + args.num_VMs * args.num_times) + args.num_PMs * args.num_resources,
                        device=args.device)
                    for j in range(args.num_users):
                        a = Request_list[j].a
                        d = Request_list[j].d
                        e = Request_list[j].e
                        b = Request_list[j].b
                        if state[-1] >= a and state[-1] <= d - e + 1 and j not in allocated:
                            S = np.zeros((args.num_VMs, args.num_times))
                            for k in range(args.num_VMs):
                                for t in range(Request_list[j].e):
                                    S[k, t] = Request_list[j].S[k][ t]
                            time_array = np.array([a, d, e, b])
                            S_time_array = np.hstack((time_array, S.reshape(-1)))
                            state[j * (4 + args.num_VMs * args.num_times):(j + 1) * (
                                        4 + args.num_VMs * args.num_times)] = torch.tensor(S_time_array,device=args.device)

                    state[args.num_users * (4 + args.num_VMs * args.num_times):args.num_users * (
                                4 + args.num_VMs * args.num_times) + args.num_PMs * args.num_resources] = \
                    wait_allocate_resources[int(state[-1] - 1)].reshape(-1)

                    state[-1] += 1


            pay = pay.sum(-1)
            ma = (pay_bids > 0).float()
            pays = pay[:args.num_users] - pay[-1] + pay_bids
            ma2 = (pays <= pay_bids).float()
            ma3 = (pays > 0).float()
            pays = pays * ma*ma2*ma3+ma*ma3*(1-ma2)*pay_bids
            paysss = pays.sum() + paysss
        paysss = paysss / args.test_num_envs
        win_num = win_num / args.test_num_envs
        welfare = welfare / args.test_num_envs
        print(f"pay{paysss}, win num{win_num}, welfare{welfare}")
