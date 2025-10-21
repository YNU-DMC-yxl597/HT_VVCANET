import torch
from torch import nn
from torch.nn import functional as F



    
    
""" ActorCritic Network """
class ActorCriticNetworkBundle(nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_units,args):
        
        super().__init__()

        self.N = args.num_users
        self.num_VMs = args.num_VMs
        self.num_PMs = args.num_PMs
        self.num_resources = args.num_resources
        self.T = args.num_times

        fc_net = []

        last_layer_dim =  self.N * (4+self.num_VMs*self.T)+ self.num_PMs*self.num_resources+self.num_VMs*self.num_resources + 1
        for _ in range(num_hidden_layers):
            fc_net.append(nn.Linear(last_layer_dim, num_hidden_units))
            fc_net.append(nn.Tanh())
            last_layer_dim = num_hidden_units

        self.fc_critic = nn.Linear(num_hidden_units, 1)
        self.fc_actor =nn.Linear(last_layer_dim,  self.N*2)
        self.fc_shared=nn.Sequential(*fc_net)

    def forward(self, x):
        x = self.fc_shared(x)
        return x
    def get_value(self, x):

        x=self.forward(x)
        value = self.fc_critic(x)
        return value

    def get_action(self, x):
        x=self.forward(x)
        action=self.fc_actor(x)

        action_x_ij = action.reshape(-1, self.N,2)
        action_x_ij=F.softmax(action_x_ij,dim=-1)

        dist_discrete = torch.distributions.Categorical(probs=action_x_ij)
        action_discrete = dist_discrete.sample()  # shape [batch_size, m]
        log_prob_discrete = dist_discrete.log_prob(action_discrete)

        return action_discrete, log_prob_discrete.sum(-1)

    def get_action_and_value(self, x):


        action,log_prob= self.get_action(x)
        value = self.get_value(x)

        return action,log_prob, value

   