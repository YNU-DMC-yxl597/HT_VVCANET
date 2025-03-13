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

    X_final=np.zeros((2**N,N,T))
    X_jilu=np.zeros(2**N)
    X_remain=np.zeros((L,N,T))
    X_resource = C.reshape(1,1,P,R).repeat(T,axis=1).repeat(L,axis=0)
    X_final_N = np.zeros((L, N, T))
    X_jilu[0]=1
    new_x=0
    insert_x=0
    for i in range(n_iter):
        selected_option=random.randint(0,L-1)
        X_temp=copy.deepcopy(X_remain[selected_option])
        C_re=copy.deepcopy(X_resource[selected_option])
        for selected_user in range(N):
            if X_temp[selected_user].sum()<1:
                for o in range(Request_list[selected_user].a-1,Request_list[selected_user].d-Request_list[selected_user].e+1):
                    X_temp[selected_user,o]=1
                    flag = 0
                    C_resource = copy.deepcopy(C_re)
                    for j in range(o,o+Request_list[selected_user].e):
                        C_temp = copy.deepcopy(C_re[j])
                        for q in range(K):
                            flag_i = 0
                            if Request_list[selected_user].S[q][j-o]==0:
                                flag_i=1

                            for number in range(Request_list[selected_user].S[q][j-o]):
                                p_num=0
                                p = random.randint(0,P-1)
                                while p_num<P:
                                    # r_i = 0
                                    flag_i_p = 0
                                    C_temp_i = copy.deepcopy(C_temp)
                                    for r in range(R):
                                        # a = Request_list[k].S[q][j - int(X_t[k].item())] * Q[q][r]
                                        r_i =  Q[q][r]
                                        if C_temp[p, r] < r_i:
                                            flag_i_p = 1
                                            flag_i=0
                                            break
                                        else:
                                            C_temp_i[p, r] -= r_i
                                    p=(p+1)%P
                                    if flag_i_p == 0:
                                        flag_i = 1
                                        C_temp = C_temp_i
                                        break
                                    p_num += 1
                                if flag_i == 0:
                                    break
                            if flag_i == 0:
                                flag=1
                                break
                        if flag==1:
                            break
                        C_resource[j] = copy.deepcopy(C_temp)
                    if flag == 0:
                        a = X_final_N.sum(-1)
                        a = np.array(a)
                        X_final[int((X_temp.sum(-1)*weight_i).sum())] =X_temp
                        if X_jilu[int((X_temp.sum(-1) * weight_i).sum())] == 0:
                            X_final_N[insert_x]=X_temp
                            insert_x = (insert_x+1)%L
                        X_jilu[int((X_temp.sum(-1) * weight_i).sum())] = 1
                        X_resource[new_x] = copy.deepcopy(C_resource)
                        X_remain[new_x] = X_temp
                        new_x = (new_x+1)%L
                        X_temp[selected_user,o] = 0
                    else:
                        X_temp[selected_user,o] =0

    X_final=X_final.sum(-1)
    X_final_N = X_final_N.sum(-1)
    return X_final_N


def generate_menu(N,K,P,R,Q,C,Request_list,T,num,L,n_iter):
    menus=np.zeros([num,L,N])
    weight_i = np.ones(N)
    for i in range(1, N):
        weight_i[i] = weight_i[i - 1] * 2
    for i in range(num):
        print(i)
        X_final=caculate_menu(N,K,P,R,Q[i],C[i],Request_list[i],T,L,n_iter,weight_i)
        menus[i]=X_final
    return menus

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
    # 用户数量
    N = 8
    # vm数量
    K = 5
    # PM数量
    P = 3
    # 资源种类
    R = 3
    #最大时间
    Tmax=5

    seed = 2024
    set_seed(seed)


    train_num=960
    test_num=960

    #菜单
    L=200
    n_iter=200

    path='data/'+str(N)+'x'+str(K)+'x'+str(P)+'x'+str(R)+'x'+str(Tmax)+'/train'
    train_Q,train_C,train_Request_list=load_data(path)
    start = time.time()
    train_menus=generate_menu(N,K,np.array(train_C).shape[1],R,train_Q,train_C,train_Request_list,Tmax,train_num,L,n_iter)
    end = time.time()
    print(str(end - start))
    np.save(os.path.join(path, "menus"), train_menus, allow_pickle=True, fix_imports=True)

    start = time.time()
    path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'x'+str(R)+'x'+str(Tmax)+'/test'
    test_Q, test_C, test_Request_list = load_data(path)
    test_menus=generate_menu(N, K, P, R, test_Q, test_C, test_Request_list, Tmax, test_num,L,n_iter)
    end = time.time()
    print(str(end - start))
    np.save(os.path.join(path, "menus"), test_menus, allow_pickle=True, fix_imports=True)




