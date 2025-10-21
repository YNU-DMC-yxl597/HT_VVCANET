import itertools
import numpy as np
import torch
import os
class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b
""" Envs and Distrs"""
class VDistBase:
    def __init__(self, args,device = "cuda"):

        self.num_users = args.num_users
        self.num_VMs = args.num_VMs
        self.num_PMs = args.num_PMs
        self.num_resources = args.num_resources
        self.num_times = args.num_times
        self.device = device

        path = 'F:/yanjiusheng/shixuyigou/data/' + str(self.num_users) + 'x' + str(self.num_VMs) + 'x' + str(self.num_PMs) + 'x' + str(self.num_resources) + 'x' + str(self.num_times) + '/train'
        self.train_Q, self.train_C, self.train_Request_list= self.load_data(path)

        self.train_user_time=np.zeros((len(self.train_Request_list),self.num_users,3))
        for i in range(len(self.train_Request_list)):
            for j in range(self.num_users):
                self.train_user_time[i,j,0]=self.train_Request_list[i,j].a
                self.train_user_time[i,j,1]=self.train_Request_list[i,j].d
                self.train_user_time[i, j, 2] = self.train_Request_list[i, j].e

        self.train_user_bids = np.zeros((len(self.train_Request_list), self.num_users))
        for i in range(len(self.train_Request_list)):
            for j in range(self.num_users):
                self.train_user_bids[i, j] = self.train_Request_list[i, j].b

        self.train_user_VMs= np.zeros((len(self.train_Request_list), self.num_users,self.num_VMs,self.num_times))
        for i in range(len(self.train_Request_list)):
            for j in range(self.num_users):
                for k in range(self.num_VMs):
                    for t in range(self.train_Request_list[i, j].e):
                        self.train_user_VMs[i, j, k,t] = self.train_Request_list[i, j].S[k][t]



        path = 'F:/yanjiusheng/shixuyigou/data/' + str(self.num_users) + 'x' + str(self.num_VMs) + 'x' + str(self.num_PMs) + 'x' + str(self.num_resources) + 'x' + str(self.num_times) + '/test'
        self.test_Q, self.test_C, self.test_Request_list = self.load_data(path)

        self.test_user_time = np.zeros((len(self.test_Request_list), self.num_users, 3))
        for i in range(len(self.test_Request_list)):
            for j in range(self.num_users):
                self.test_user_time[i, j, 0] = self.test_Request_list[i, j].a
                self.test_user_time[i, j, 1] = self.test_Request_list[i, j].d
                self.test_user_time[i, j, 2] = self.test_Request_list[i, j].e

        self.test_user_bids = np.zeros((len(self.test_Request_list), self.num_users))
        for i in range(len(self.test_Request_list)):
            for j in range(self.num_users):
                self.test_user_bids[i, j] = self.test_Request_list[i, j].b

        self.test_user_VMs = np.zeros((len(self.test_Request_list), self.num_users, self.num_VMs,self.num_times))
        for i in range(len(self.test_Request_list)):
            for j in range(self.num_users):
                for k in range(self.num_VMs):
                    for t in range(self.test_Request_list[i, j].e):
                        self.test_user_VMs[i, j, k,t] = self.test_Request_list[i, j].S[k][t]


    def load_data(self,dir):
        Q = np.load(os.path.join(dir, 'Q.npy'))
        C = np.load(os.path.join(dir, 'C.npy'))
        Request_list = np.load(os.path.join(dir, 'Request_list.npy'), allow_pickle=True)
        return Q, C, Request_list

    def sample_state(self,ind_i,train=True):
        if train:
            return self.train_user_bids[ind_i],self.train_user_time[ind_i,:,:],self.train_user_VMs[ind_i,:,:]
        else:
            return self.test_user_bids[ind_i],self.test_user_time[ind_i,:,:],self.test_user_VMs[ind_i,:,:]

    def sample(self,ind_i,train=True):
        if train:
            return self.train_Request_list[ind_i],self.train_Q[ind_i,:self.num_VMs],self.train_C[ind_i]
        else:
            return self.test_Request_list[ind_i],self.test_Q[ind_i,:self.num_VMs],self.test_C[ind_i]

        
        