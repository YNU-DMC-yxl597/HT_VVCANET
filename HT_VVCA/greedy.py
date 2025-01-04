import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import generate_data
import torch.nn.functional as F
import numpy as np
import os
import copy
import time
class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b

def alloc(N,K,P,R,Q,C,Request_list,T):
    index_user=np.arange(N,dtype=int)
    density = []
    for i in range(N):
        vm_num=0
        for j in range(K):
            for t in range(Request_list[i].e):
                vm_num+=Request_list[i].S[j][t]
        if vm_num==0:
            density.append(1000)
        else:
            density.append(Request_list[i].b/vm_num)
    density = np.array(density)
    for i in range(N):
        for j in range(0,N-i-1):
            if density[j]<density[j+1]:
                temp=density[j+1]
                density[j+1]=density[j]
                density[j]=temp
                temp=index_user[j+1]
                index_user[j+1]=index_user[j]
                index_user[j]=temp
    X_full = np.zeros(N)
    X_temp_full = np.zeros((N,T))
    for i in range(N):
        flag = 0
        select_i=index_user[i]
        X_temp_full[select_i,Request_list[select_i].a-1]=1
        for t in range(Request_list[select_i].e):
            X_temp_full[select_i, t + Request_list[select_i].a-1] = 1
        temp_Request_list=copy.deepcopy(Request_list)
        for j in range(T):
            C_temp = copy.deepcopy(C)
            for k in range(N):
                if X_temp_full[k, j] == 1:
                    for q in range(K):
                        flag_i = 1
                        for o in range(Request_list[k].S[q][j - int(Request_list[k].a-1)]):
                            flag_i=0
                            for p in range(P):
                                # r_i = 0
                                flag_i_p = 0
                                C_temp_i = copy.deepcopy(C_temp)
                                for r in range(R):
                                    # a = Request_list[k].S[q][j - int(X_t[k].item())] * Q[q][r]
                                    r_i = Q[q][r]
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
        if flag == 1:
            X_temp_full[select_i]=np.zeros(T)
        else:
            X_full[select_i]=1
    return X_full

def payment(N,K,P,R,Q,C,Request_list,T,thr,alloc_X):
    payments=np.zeros(N,dtype=float)
    welfare=0
    num_win=0
    for i in range(N):
        if alloc_X[i]==1:
            welfare+=Request_list[i].b
            num_win+=1
            l=0
            h=Request_list[i].b
            while h-l>thr:
                temp_Request_list=copy.deepcopy(Request_list)
                temp_Request_list[i].b=(l+h)/2
                X_full=alloc(N,K,P,R,Q,C,temp_Request_list,T)
                if X_full[i]==1:
                    h=(l+h)/2
                else:
                    l=(l+h)/2
            payments[i]=h
        else:
            payments[i]=0
    return payments,welfare,num_win
def load_data(dir):
    Q=np.load(os.path.join(dir, 'Q.npy'))
    C=np.load(os.path.join(dir, 'C.npy'))
    Request_list=np.load(os.path.join(dir, 'Request_list.npy'),allow_pickle=True)

    return Q,C,Request_list
def set_seed(seed):
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
    P = 5
    R = 3
    Tmax=3

    seed = 2024
    set_seed(seed)


    train_num=1000
    test_num=960
    batch_size=64

    train_seed=2023
    test_seed=2024

    thr=0.001


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cuda'

    path='data/'+str(N)+'x'+str(K)+'x'+str(P)+'/train'

    path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'/test'
    test_Q, test_C, test_Request_list = load_data(path)

    print(np.array(test_C).shape)
    P = np.array(test_C).shape[1]

    test_pay_sum = 0
    welfare_sum=0
    num_win_sum=0
    start = time.time()
    for i in range(test_num):
        alloc_X = alloc(N, K, P, R, test_Q[i], test_C[i], test_Request_list[i], Tmax)
        payments,welfare,num_win = payment(N, K, P, R, test_Q[i], test_C[i], test_Request_list[i], Tmax, thr, alloc_X)
        test_pay_sum += payments.sum()
        welfare_sum+=welfare
        num_win_sum+=num_win
    print("平均收入是{}".format(test_pay_sum / test_num))
    print("社会福利{}".format(welfare_sum/test_num))
    print("获胜对{}".format(num_win_sum/test_num))
    end = time.time()
    print(str(end - start))
