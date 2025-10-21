import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box
import copy
import random
class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b
class AuctionEnv(Env):
    def __init__(self,args,v_dist,ind_i,mask_N,train):

        self.num_users = args.num_users
        self.num_VMs = args.num_VMs
        self.num_PMs = args.num_PMs
        self.num_resources = args.num_resources
        self.num_times = args.num_times
        self.ind_i = ind_i


        self.sample = v_dist.sample
        self.sample_state = v_dist.sample_state

        self.train=train

        self.action_space = Box(low = 0, high = 1.0, shape=(self.num_users,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=np.inf,
                                     shape=(self.num_users * (4+self.num_VMs*self.num_times)+ self.num_PMs*self.num_resources+self.num_VMs*self.num_resources + 1,),
                                     dtype=np.float32)
        self.mask_N = mask_N
        self.reset()

    def preprocess_action(self, action):
        reward=0
        for i in range(self.num_users):
            flag = 0
            a = self.Request_list[i].a
            d = self.Request_list[i].d
            e = self.Request_list[i].e
            b = self.Request_list[i].b
            if action[i]==1 and self.state[-1]>=a and self.state[-1]<=d-e+1 and i not in self.allocated and i!=self.mask_N:
                C_temp_times=copy.deepcopy(self.wait_allocate_resources)
                for j in range(int(self.state[-1])-1,int(self.state[-1])-1+e):
                    C_temp = copy.deepcopy(C_temp_times[j])
                    for q in range(self.num_VMs):
                        flag_i = 1
                        for o in range(self.Request_list[i].S[q][j - int(self.state[-1] - 1)]):
                            flag_i = 0
                            for p in range(self.num_PMs):
                                flag_i_p = 0
                                C_temp_i = copy.deepcopy(C_temp)
                                for r in range(self.num_resources):
                                    r_i = self.Q[q][r]
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
                    self.wait_allocate_resources = C_temp_times
                    self.allocated.append(i)
                    reward = reward + b

        self.state[:self.num_users * (
                    4 + self.num_VMs * self.num_times) + self.num_PMs * self.num_resources] = np.zeros(self.num_users * (
                    4 + self.num_VMs * self.num_times) + self.num_PMs * self.num_resources, dtype=np.float32)
        for j in range(self.num_users):
            a=self.Request_list[j].a
            d=self.Request_list[j].d
            e = self.Request_list[j].e
            b=self.Request_list[j].b
            if self.state[-1]>=a and self.state[-1]<=d-e+1 and j not in self.allocated:
                S=np.zeros((self.num_VMs,self.num_times))
                for k in range(self.num_VMs):
                    for t in range(self.Request_list[j].e):
                        S[k,t] = self.Request_list[j].S[k][t]
                time_array=np.array([a,d,e,b])
                S_time_array=np.hstack((time_array,S.reshape(-1)))
                self.state[j*(4+self.num_VMs*self.num_times):(j+1)*(4+self.num_VMs*self.num_times)]=S_time_array


        self.state[self.num_users * (4+self.num_VMs*self.num_times):self.num_users * (4+self.num_VMs*self.num_times)+ self.num_PMs*self.num_resources]=self.wait_allocate_resources[int(self.state[-1]-1)].reshape(-1)

        self.state[-1]+=1

        return reward
    
        
    def step(self, action):
        """ Preprocess action """
        reward=self.preprocess_action(action)

        terminated = (int(self.state[-1]) == self.num_times+1)

        info = {}
        
        return self.state, reward, terminated, False,info

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)


        self.state = np.zeros(self.num_users * (4+self.num_VMs*self.num_times)+ self.num_PMs*self.num_resources+self.num_VMs*self.num_resources + 1, dtype=np.float32)
        self.state[-1] = 1.0
        self.Request_list, self.Q, self.C = self.sample(self.ind_i,self.train)

        self.allocated=[]
        for j in range(self.num_users):
            a=self.Request_list[j].a
            d=self.Request_list[j].d
            e = self.Request_list[j].e
            b=self.Request_list[j].b
            if self.state[-1]>=a and self.state[-1]<=d-e+1 and j not in self.allocated  and j!=self.mask_N:
                S=np.zeros((self.num_VMs,self.num_times))
                for k in range(self.num_VMs):
                    for t in range(self.Request_list[j].e):
                        S[k,t] = self.Request_list[j].S[k][t]
                time_array=np.array([a,d,e,b])
                S_time_array=np.hstack((time_array,S.reshape(-1)))
                self.state[j*(4+self.num_VMs*self.num_times):(j+1)*(4+self.num_VMs*self.num_times)]=S_time_array


        self.state[self.num_users * (4+self.num_VMs*self.num_times):self.num_users * (4+self.num_VMs*self.num_times)+ self.num_PMs*self.num_resources]=self.C.reshape(-1)
        self.state[self.num_users * (4+self.num_VMs*self.num_times)+ self.num_PMs*self.num_resources:-1] = self.Q.reshape(-1)

        self.wait_allocate_resources=np.repeat(np.expand_dims(np.array(self.C), axis=0),self.num_times,axis=0)

        return self.state, {}

