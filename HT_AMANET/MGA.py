import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy
import random
import time
class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b


def caculate_menu(N,K,P,R,Q,C,Request_list,T,L,n_iter,weight_i):

    X_final=torch.zeros((L,N,T))
    X_jilu=torch.zeros(L)
    X_jilu[0]=1
    new_x=0
    insert_x=0
    for i in range(n_iter):
        j=random.randint(0,X_final.shape[0]-1)
        X_temp=copy.deepcopy(X_final[j])
        for j in range(N):
            k=random.randint(0,N)
            if k==0:
                if X_temp[j].sum()>=1:
                    X_temp[j]=torch.zeros(T)
                else:
                    t=random.randint(Request_list[j].a-1,Request_list[j].d-Request_list[j].e)
                    X_temp[j,t]=1
        if X_jilu[int((X_temp.sum(-1)*weight_i).sum())]==0:
            X_t = torch.zeros(N)
            flag = 0
            X_temp_full = copy.deepcopy(X_temp)
            for j in range(N):
                for t in range(T):
                    if X_temp_full[j, t] == 1:
                        break
                if X_temp_full[j, t] == 1:
                    X_t[j] = t
                    for k in range(Request_list[j].e):
                        X_temp_full[j, t + k] = 1
            for j in range(T):
                C_temp = copy.deepcopy(C)
                for k in range(N):
                    if X_temp_full[k, j] == 1:
                        for q in range(K):
                            flag_i = 1

                            for o in range( Request_list[k].S[q][j - int(X_t[k].item())]):
                                flag_i = 0
                                for p in range(P):
                                    # r_i = 0
                                    flag_i_p = 0
                                    C_temp_i = copy.deepcopy(C_temp)
                                    for r in range(R):
                                        r_i =  Q[q][r]
                                        if C_temp[p, r] < r_i:
                                            flag_i_p = 1
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
                if flag == 1:
                    break
            if flag == 0:
                X_final[int((X_temp.sum(-1)*weight_i).sum())] =X_temp
                X_jilu[int((X_temp.sum(-1) * weight_i).sum())] = 1

    X_final=X_final.sum(-1)
    X_final=np.array(X_final)
    return X_final


def generate_menue(N,K,P,R,Q,C,Request_list,T,num,L,n_iter):
    menues=np.zeros([num,L,N])
    weight_i = torch.ones(N)
    for i in range(1, N):
        weight_i[i] = weight_i[i - 1] * 2
    for i in range(num):
        print(i)
        X_final=caculate_menu(N,K,P,R,Q[i],C[i],Request_list[i],T,L,n_iter,weight_i)
        menues[i]=X_final
    return menues

def load_data(dir):
    Q=np.load(os.path.join(dir, 'Q.npy'))
    C=np.load(os.path.join(dir, 'C.npy'))
    Request_list=np.load(os.path.join(dir, 'Request_list.npy'),allow_pickle=True)

    return Q,C,Request_list

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    N = 4
    K = 10
    P = 2
    R = 3
    Tmax=3

    seed = 2024
    set_seed(seed)


    train_num=5000
    test_num=960
    batch_size=64

    train_seed=2023
    test_seed=2024

    hidden_size=128
    L=2**N
    n_iter=L*20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cuda'

    path='data/'+str(N)+'x'+str(K)+'x'+str(P)+'/train'
    train_Q,train_C,train_Request_list=load_data(path)
    start = time.time()
    train_menues=generate_menue(N,K,np.array(train_C).shape[1],R,train_Q,train_C,train_Request_list,Tmax,train_num,L,n_iter)
    end = time.time()
    print(str(end - start))
    np.save(os.path.join(path, "menus"), train_menues, allow_pickle=True, fix_imports=True)

    start = time.time()
    path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'/test'
    test_Q, test_C, test_Request_list = load_data(path)
    test_menues=generate_menue(N, K, P, R, test_Q, test_C, test_Request_list, Tmax, test_num,L,n_iter)
    end = time.time()
    print(str(end - start))
    np.save(os.path.join(path, "menus"), test_menues, allow_pickle=True, fix_imports=True)




