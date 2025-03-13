import numpy as np
import copy
import cplex_solve_problem
import os
import time
import sys
print(sys.version)
class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b

def caculate_revenue(N,K,P,R,Q,C,Request_list,T,w,lam):
    revenue = 0
    temp_Request_list = copy.deepcopy(Request_list)

    for i in range(N):
        temp_Request_list[i].b = w[i] * Request_list[i].b + lam[i]
    welfare,solution_x=cplex_solve_problem.cplex_solve_question(N,K,P,R,Q,C,temp_Request_list,T)
    for i in range(N):
        flag=0
        for j in range(T):
            if solution_x[i*T+j]==1:
                flag=1
                break
        if flag==1:
            Request_list_i=temp_Request_list.copy()
            del Request_list_i[i]
            welfare_i,_=cplex_solve_problem.cplex_solve_question(N-1,K,P,R,Q,C,Request_list_i,T)
            revenue=revenue+(welfare_i-welfare+w[i]*Request_list[i].b)/w[i]

    num_win=0
    welfare_sum=0
    for i in range(N):
        for j in range(T):
            if solution_x[i*T+j]==1:
                num_win+=1
                welfare_sum+=Request_list[i].b

    return revenue,num_win,welfare_sum

def load_data(dir):
    Q=np.load(os.path.join(dir, 'Q.npy'))
    C=np.load(os.path.join(dir, 'C.npy'))
    Request_list=np.load(os.path.join(dir, 'Request_list.npy'),allow_pickle=True)

    return Q.tolist(),C.tolist(),Request_list.tolist()

def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    # 用户数量
    N = 10
    # vm数量
    K = 5
    # PM数量
    P = 3
    # 资源种类
    R = 3
    # 最大时间
    Tmax = 5

    seed = 2024
    set_seed(seed)

    train_num = 960
    test_num = 960

    path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'x'+str(R)+'x'+str(Tmax)+'/train'
    train_Q, train_C, train_Request_list= load_data(path)
    path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'x'+str(R)+'x'+str(Tmax)+'/test'
    test_Q, test_C, test_Request_list= load_data(path)

    w=np.ones(N)
    lam=np.zeros(N)
    P=np.array(test_C).shape[1]

    revenue_sum=0
    num_win_sum=0
    welfare_sum=0

    start = time.time()

    for l in range(test_num):
        Q = test_Q[l]
        C = test_C[l]
        Request_list = test_Request_list[l]
        revenue, num_win, welfare=caculate_revenue(N, K, P, R, Q, C, Request_list, Tmax, w, lam)
        revenue_sum += revenue
        num_win_sum += num_win
        welfare_sum+=welfare

    print("平均收入是{},获胜对{}，社会福利{}".format(revenue_sum/test_num,num_win_sum/test_num,welfare_sum/test_num))
    end = time.time()
    print(str(end - start))






