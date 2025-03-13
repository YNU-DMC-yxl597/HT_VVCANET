import numpy as np
import os
class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b
def generate_data(sample_num,N,K,P,R,T,seed):
    np.random.seed(seed)
    Q = (np.repeat([[[1, 2,3], [3, 4,2], [0, 1,1], [1, 1,0], [1, 0,1]]], sample_num, axis=0)).tolist()
    C=(np.ones([sample_num,P,R])*5).tolist()


    Request_list = []
    for j in range(sample_num):
        Request_list_i = []
        for i in range(N):
            a=int(np.random.randint(1,T+1,size=1)[0])
            d=int(np.random.randint(a,T+1,size=1)[0])
            e=int(np.random.randint(1,d-a+2,size=1)[0])
            S = np.zeros([K, e], dtype=int)
            for t in range(e):
                s = np.random.randint(0, 3, [K])
                while s.sum() == 0:
                        s = np.random.randint(0, 3, [K])
                S[:,t] = s
            b = float(np.random.normal(1, 0.1))
            while b < 0:
                b=float(np.random.normal(1, 0.1))
            b = b * (S.sum())
            S=S.tolist()
            Request_list_i.append(Request(a,d,S,e,b))
        Request_list.append(Request_list_i)

    T=[]
    for j in range(sample_num):
        T.append(max([Request_list[j][i].d for i in range(N)]))
    return Q,C,Request_list,T
if __name__ == '__main__':
    # 用户数量
    N = 8
    # vm数量
    K = 5
    # PM数量
    P = 3
    # 资源种类
    R = 3
    # 最大时间
    Tmax = 5

    train_num = 5000
    test_num = 1000

    train_seed = 2024
    test_seed = 1111


    train_Q, train_C, train_Request_list, train_T = generate_data(train_num, N, K, P, R, Tmax, train_seed)
    test_Q, test_C, test_Request_list, test_T = generate_data(test_num, N, K, P, R, Tmax, test_seed)

    path = os.path.join("data", str(N) + 'x' + str(K)+ 'x' + str(P)+'x'+str(R)+'x'+str(Tmax))
    if not os.path.exists(path):
        os.makedirs(path)


    path_dir = os.path.join(path, "train")
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    np.save(os.path.join(path_dir, "Q"), train_Q, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "C"), train_C, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "Request_list"), train_Request_list, allow_pickle=True, fix_imports=True)

    path_dir = os.path.join(path, "test")
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    np.save(os.path.join(path_dir, "Q"), test_Q, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "C"), test_C, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "Request_list"), test_Request_list, allow_pickle=True, fix_imports=True)

