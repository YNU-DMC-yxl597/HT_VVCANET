import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
import time
import csv
import codecs

class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b

class net(nn.Module):
    def __init__(self, hidden_size,N,K,P,R,Tmax,L,batch_size,softmax_temp,device):
        super(net, self).__init__()
        self.mlp1=nn.Sequential(nn.Linear(N, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.TransformerEncoderLayer(hidden_size, 4, hidden_size, batch_first=True, dropout=0),

                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1)
                                )
        self.n_bidder_representation=torch.arange(N).repeat(batch_size).reshape(batch_size, N).long()
        self.bidder_embeddings = nn.Embedding(N, hidden_size)
        self.wnet=nn.Sequential(
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, N)
        )

        self.N = N
        self.K = K
        self.P = P
        self.R = R
        self.Tmax = Tmax
        self.L = L
        self.batch_size = batch_size
        self.softmax_temp = softmax_temp
        self.device = device

        mask = 1 - torch.eye((self.N)).to(self.device)
        self.mask = torch.zeros(self.N, self.batch_size, self.N).to(self.device)
        for i in range(self.N):
            self.mask[i] = mask[i].repeat(self.batch_size, 1)
        self.mask = self.mask.reshape(self.N * self.batch_size, self.N)

    def forward(self,Request_list,menues):

        bid = []
        for l in range(self.batch_size):
            for i in range(self.N):
                bid.append(Request_list[l][i].b)
        bid = torch.tensor(bid).to(self.device)

        menue_item = torch.tensor(menues, device=device)
        menue_item = menue_item.float()

        bid_vector= bid.reshape(-1,1,self.N).repeat(1,self.L,1).permute(0,2,1).float()
        input_vector = menue_item
        output = self.mlp1(input_vector)

        w = self.n_bidder_representation.to(self.device).float()+1
        w = self.wnet(w)
        w = torch.sigmoid(w)
        lamb=output[:, :, 0]

        bid = bid.reshape(self.batch_size, 1,self.N)

        allocs = menue_item
        util_from_items = allocs * bid  # B, t, n
        per_agent_welfare = w.unsqueeze(1) * util_from_items  # B, t, n
        total_welfare = per_agent_welfare.sum(axis=-1)  # B, t
        alloc_choice = F.softmax((total_welfare + lamb) * self.softmax_temp, dim=-1)  # B, t
        chosen_alloc_welfare_per_agent = (per_agent_welfare * torch.unsqueeze(alloc_choice, -1)).sum(axis=1)  # B, n

        welfare = (util_from_items * torch.unsqueeze(alloc_choice, -1)).sum(axis=1)
        welfare = welfare.sum()

        if N > 1:

            n_chosen_alloc_welfare_per_agent = chosen_alloc_welfare_per_agent.repeat(self.N, 1)  # nB, n
            masked_chosen_alloc_welfare_per_agent = n_chosen_alloc_welfare_per_agent * self.mask  # nB, n
            n_per_agent_welfare = per_agent_welfare.repeat(self.N, 1, 1)  # nB, t, n
            removed_i_welfare = n_per_agent_welfare * self.mask.reshape(self.N * self.batch_size, 1, self.N)  # nB, t, n
            total_removed_welfare = removed_i_welfare.sum(axis=-1)  # nB, t
            removed_alloc_choice = F.softmax((total_removed_welfare + lamb.repeat(self.N, 1)) * self.softmax_temp, dim=-1)
            # nB, t
            removed_chosen_welfare_per_agent = (
                    removed_i_welfare * removed_alloc_choice.unsqueeze(-1)  # nB, t, n
            ).sum(axis=1)
            # nB, n
            payments = torch.zeros(self.N * self.batch_size).to(self.device)
            payments = (1 / w.permute(1, 0).reshape(self.N * self.batch_size)) * (
                    removed_chosen_welfare_per_agent.sum(-1)
                    + (removed_alloc_choice * lamb.repeat(self.N , 1)).sum(-1)
                    - masked_chosen_alloc_welfare_per_agent.sum(-1)
                    - (alloc_choice * lamb).sum(1).repeat(self.N )
            )  # nB
            payments = payments.reshape(self.N , self.batch_size)

        return payments,w,lamb,welfare

    def test_forward(self,Request_list,menues,device):
        bid = []
        for l in range(self.batch_size):
            for i in range(self.N):
                bid.append(Request_list[l][i].b)
        bid = torch.tensor(bid).to(self.device)

        menue_item = torch.tensor(menues, device=device)
        menue_item = menue_item.float()

        bid_vector = bid.reshape(-1, 1, self.N).repeat(1, self.L, 1).permute(0,2,1).float()
        input_vector = menue_item
        output = self.mlp1(input_vector)

        w=self.n_bidder_representation.to(self.device).float()+1
        w=self.wnet(w)
        w = torch.sigmoid(w)
        lamb=output[:, :, 0]

        bid = bid.reshape(self.batch_size,  1,self.N)
        allocs = menue_item
        util_from_items = allocs * bid  # B, t, n
        per_agent_welfare = w.unsqueeze(1) * util_from_items  # B, t, n
        total_welfare = per_agent_welfare.sum(axis=-1)  # B, t
        alloc_choice_ind = torch.argmax(total_welfare + lamb, -1)  # B

        chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in range(batch_size)]
        chosen_alloc_welfare_per_agent = torch.stack(chosen_alloc_welfare_per_agent)  # B, n

        welfare=[util_from_items[i, alloc_choice_ind[i], ...] for i in range(batch_size)]
        welfare=torch.stack(welfare)
        welfare=welfare.sum()

        num_win=[allocs[i, alloc_choice_ind[i], ...] for i in range(batch_size)]
        num_win=torch.stack(num_win)
        num_win=num_win.sum()

        payments = []
        for i in range(N):
            mask = torch.ones(N).to(device)
            mask[i] = 0
            removed_i_welfare = per_agent_welfare * mask.reshape(1, 1, N)
            total_removed_welfare = removed_i_welfare.sum(-1)  # B, t

            removed_alloc_choice_ind = torch.argmax(total_removed_welfare + lamb, -1)  # B
            removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in
                                      range(batch_size)]  # B
            removed_chosen_welfare = torch.stack(removed_chosen_welfare)

            removed_alloc_b = [lamb[i, removed_alloc_choice_ind[i]] for i in range(batch_size)]
            removed_alloc_b = torch.stack(removed_alloc_b)

            alloc_b = [lamb[i, alloc_choice_ind[i]] for i in range(batch_size)]
            alloc_b = torch.stack(alloc_b)

            payments.append(
                (1.0 / w[:, i])
                * (
                        (
                                removed_chosen_welfare
                                + removed_alloc_b
                        )
                        - (chosen_alloc_welfare_per_agent.sum(1) - chosen_alloc_welfare_per_agent[:, i])
                        - alloc_b
                )
            )
        payments = torch.stack(payments)
        return payments,w,lamb,welfare,num_win


def load_data(dir):
    Q=np.load(os.path.join(dir, 'Q.npy'))
    C=np.load(os.path.join(dir, 'C.npy'))
    Request_list=np.load(os.path.join(dir, 'Request_list.npy'),allow_pickle=True)
    menues=np.load(os.path.join(dir, 'menus.npy'))

    return Q,C,Request_list,menues


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_write_csv(file_name, datas): 
    file_csv = codecs.open(file_name, 'w', 'utf-8')
    writer = csv.writer(file_csv)
    for data in datas:
        if len(data)>=1:
            writer.writerow([data[0],data[1],data[2]])
        else:
            writer.writerow(data)
    print("保存文件成功，处理结束")

if __name__ == '__main__':
    N =4
    K =10
    P =2
    R = 3
    Tmax=3


    seed = 2022
    set_seed(seed)

    train_num=5000
    test_num=960
    batch_size=64

    train_seed=2023
    test_seed=2024

    hidden_size=32
    L=2**N

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cuda'

    path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'/train'
    train_Q,train_C,train_Request_list,train_menues=load_data(path)
    path = 'data/'+str(N)+'x'+str(K)+'x'+str(P)+'/test'
    test_Q, test_C, test_Request_list, test_menues = load_data(path)

    print(np.array(test_C).shape)
    P = np.array(test_C).shape[1]

    train_menues=train_menues[:,:L]
    test_menues=test_menues[:,:L]

    batch_revenue=0

    softmax_temp=500


    model = net(hidden_size,N,K,P,R,Tmax,L,batch_size,softmax_temp,device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_iter = 0
    num_epoch=100

    lossData = [[]]

    max_sw=0
    max_revenue=0
    max_win=0
    max_value=0
    for epoch in range(num_epoch):
        loss_sum=0
        w_sum=0
        lambda_sum=0
        welfare_sum=0
        DEVICE = 'cuda'
        model = model.to(DEVICE)
        model.device = DEVICE
        for j in range(int(train_num/batch_size)):
            optimizer.zero_grad()
            payment,w,lamb,welfare= model(train_Request_list[j * batch_size:(j + 1) * batch_size],train_menues[j * batch_size:(j + 1) * batch_size])

            loss=- payment.sum(0).mean()

            loss_sum+=payment.sum().item()
            w_sum+=w
            lambda_sum+=lamb
            welfare_sum+=welfare

            loss.backward()
            optimizer.step()


        print("第{}轮:{},社会福利{}".format(epoch+1,loss_sum/(int(train_num/batch_size)*batch_size),welfare_sum/(int(train_num/batch_size)*batch_size)))


        DEVICE = 'cpu'
        model = model.to(DEVICE)
        model.device = DEVICE


        with torch.no_grad():
            start = time.time()
            num_win_sum = 0
            revenue = 0
            welfare_sum=0
            w_bs=torch.zeros(int(test_num / batch_size)*batch_size,N).to(DEVICE)
            lamb_bs = torch.zeros(int(test_num / batch_size)*batch_size, L).to(DEVICE)
            for j in range(int(test_num / batch_size)):
                payment,w,lamb,welfare,num_win = model.test_forward(
                    test_Request_list[j * batch_size:(j + 1) * batch_size],
                    test_menues[j * batch_size:(j + 1) * batch_size],DEVICE
                    )
                w_bs[j*batch_size:(j+1)*batch_size,:]=w
                lamb_bs[j * batch_size:(j + 1) * batch_size, :] = lamb
                revenue += payment.sum().item()
                welfare_sum+=welfare
                num_win_sum+=num_win
            revenue /= int(test_num/batch_size)*batch_size
            welfare_sum/=int(test_num/batch_size)*batch_size
            num_win_sum/=int(test_num/batch_size)*batch_size
            end = time.time()
            print(str(end - start))
            print("最后收入：{}".format(revenue))
            print("社会福利:{}".format(welfare_sum))
            print("获胜对:{}".format(num_win_sum))
            if loss_sum / (int(train_num / batch_size) * batch_size) > max_value:
                max_value = loss_sum / (int(train_num / batch_size) * batch_size)
                max_sw=welfare_sum
                max_revenue=revenue
                max_win=num_win_sum


        lossData.append([epoch, loss_sum / (int(train_num / batch_size) * batch_size),revenue])

    w_bs=w_bs.numpy()
    print(w_bs)
    print(max_sw,max_revenue,max_win)
    lamb_bs=lamb_bs.numpy()
    np.save(os.path.join('F:\yanjiusheng\shixuyigou\data\loss', "w42"), w_bs, allow_pickle=True, fix_imports=True)
    np.save(os.path.join('F:\yanjiusheng\shixuyigou\data\loss', "lamb42"), lamb_bs, allow_pickle=True, fix_imports=True)
    data_write_csv('F:\yanjiusheng\shixuyigou\data\loss\lossData42.csv',lossData)
