import numpy as np
import copy
import cplex_solve_problem
import os
import time
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

    return revenue
def caculate_loss(N,K,P,R,Q,C,Request_list,T,w,lam):
    revenue = 0
    temp_Request_list = copy.deepcopy(Request_list)

    for i in range(N):
        temp_Request_list[i].b = w[i] * Request_list[i].b + lam[i]
    welfare,solution_x=cplex_solve_problem.cplex_solve_question(N,K,P,R,Q,C,temp_Request_list,T)
    num_win = 0
    welfare_sum = 0
    for i in range(N):
        flag=0
        for j in range(T):
            if solution_x[i*T+j]==1:
                num_win += 1
                welfare_sum += Request_list[i].b
                flag=1
                break
        if flag==1:
            Request_list_i=temp_Request_list.copy()
            del Request_list_i[i]
            welfare_i,_=cplex_solve_problem.cplex_solve_question(N-1,K,P,R,Q,C,Request_list_i,T)

            revenue=revenue+(welfare_i-welfare+w[i]*Request_list[i].b)/w[i]
    return revenue,welfare_sum,num_win
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

   num=5
   seed = 2024
   set_seed(seed)

   train_num = 960
   test_num = 960

   path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'x'+str(R)+'x'+str(Tmax)+'/train'
   train_Q, train_C, train_Request_list = load_data(path)
   path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'x'+str(R)+'x'+str(Tmax)+'/test'
   test_Q, test_C, test_Request_list = load_data(path)

   print(np.array(test_C).shape)
   P = np.array(test_C).shape[1]

   batch_size = 32

   w = np.ones(N)
   lam = np.zeros(N)

   mask = np.zeros(N)
   e_w = 0.1
   e_lam = 0.1
   lr =0.1
   batch_revenue = 0

   flag = 0
   max_value = 0
   max_sw = 0
   max_revenue = 0
   max_win = 0
   for epoch in range(num):
       print("第{}轮".format(epoch + 1))
       loss_sum=0
       flag=0
       train_sum=0
       for j in range(int(train_num / batch_size)):
           diff_w = np.zeros(N)
           diff_lam = np.zeros(N)
           for k in range(batch_size):
               train_sum+=1
               Q = train_Q[j * batch_size + k]
               C = train_C[j * batch_size + k]
               Request_list = train_Request_list[j * batch_size + k]
               T = Tmax

               revenue= caculate_revenue(N, K, P, R, Q, C, Request_list, T, w, lam)
               loss_sum = loss_sum + revenue
               for i in range(N):
                   mask[i] = e_w
                   diff_w[i] = (caculate_revenue(N, K, P, R, Q, C, Request_list, T, w + mask, lam) - revenue) / e_w + \
                               diff_w[i]
                   mask[i] = e_lam
                   diff_lam[i] = (caculate_revenue(N, K, P, R, Q, C, Request_list, T, w,
                                                   lam + mask) - revenue) / e_lam + diff_lam[i]
                   mask[i] = 0

           if (np.array(w + diff_w * lr / batch_size) < 0).any():
               flag = 1
               break
           w = w + diff_w * lr / batch_size
           lam = lam + diff_lam * lr / batch_size
       revenue_sum = 0
       welfare_sum=0
       num_win_sum=0
       start = time.time()
       for l in range(test_num):
           Q = test_Q[l]
           C = test_C[l]
           Request_list = test_Request_list[l]
           T = Tmax
           revenue,welfare,num_win = caculate_loss(N, K, P, R, Q, C, Request_list, T, w, lam)
           # print(revenue)
           welfare_sum += welfare
           num_win_sum += num_win
           revenue_sum+=revenue
       if loss_sum/train_sum > max_value:
           max_value = loss_sum/train_sum
           max_sw = welfare_sum/ test_num
           max_revenue = revenue_sum/ test_num
           max_win = num_win_sum/ test_num
       end = time.time()
       print(str(end - start))
       print("社会福利{}".format(welfare_sum / test_num))
       print("获胜对{}".format(num_win_sum / test_num))
       print("平均收入是{}".format(revenue_sum / test_num))

   print("社会福利{}".format(max_sw),"获胜对{}".format(max_win),"平均收入是{}".format(max_revenue))





