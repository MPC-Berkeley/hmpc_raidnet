from typing import Any
import torch as th
from torch import nn
import numpy as np
import pdb
from infrastructure.utils import observation_flatten, observation_unflatten

class mlp(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self,input_dim,hidden_size,output_dim,num_layers, lambda_dim, pred_mode):
        super(mlp ,self).__init__()
        self.num_layers = num_layers

        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = []

        for i in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size,hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.fc_hidden = nn.Sequential(*self.hidden_layers)
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.pred = nn.Sequential(self.fc_in,self.fc_hidden,self.fc_out)
        self.sigmoid = nn.Sigmoid()

        #parameters necessary for bc.py
        self.lambda_dim = lambda_dim
        self.pred_mode = pred_mode
        self.output_dim = output_dim
        self.num_layers = num_layers

  def __call__(self, state):
      '''
      state: th.Tensor
      out: th.Tensor
      '''
      out = self.sigmoid(self.pred(state))
      return out
  
class R_Tf(nn.Module):
  '''
    Recurrent Transformer arch
  '''
  def __init__(self,input_dim, embed_dim, output_dim, horizon, num_layers,hidden_size,lambda_dim = None, lambda_ubd = 1000):
        super(R_Tf ,self).__init__()
        
        self.Q_dim = [4,3]   # num_vs x [state, mode]
        self.lift=nn.Linear(self.Q_dim[1], embed_dim)
        self.mh_attn=nn.MultiheadAttention(embed_dim, 1)
        self.add_norm=nn.LayerNorm(embed_dim)

        self.num_layers = num_layers
        self.fc_in = nn.Linear(embed_dim, hidden_size)
        self.hidden_layers = []

        for i in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size,hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.fc_hidden = nn.Sequential(*self.hidden_layers)
        self.fc_out = nn.Linear(hidden_size, embed_dim)
        self.pred = nn.Sequential(self.fc_in,self.fc_hidden,self.fc_out)


        # Decoder attn
        self.N=horizon # should be SMPC.N-1       

        self.mh_attn_dc = nn.MultiheadAttention(embed_dim, 1)

        
        self.fc_in_d = nn.Linear(embed_dim, hidden_size)
        self.hidden_layers_d = []

        for i in range(num_layers):
            self.hidden_layers_d.append(nn.Linear(hidden_size,hidden_size))
            self.hidden_layers_d.append(nn.ReLU())

        self.fc_hidden_d = nn.Sequential(*self.hidden_layers)
        self.fc_out_d = nn.Linear(hidden_size, embed_dim)
        self.pred_d = nn.Sequential(self.fc_in_d,self.fc_hidden_d,self.fc_out_d)


        self.project = nn.Linear(embed_dim * self.Q_dim[0], int(output_dim/self.N) ) 

        self.rnn_d=nn.GRU(embed_dim, embed_dim, batch_first=True)
        
        #clip the output
        self.lambda_dim=lambda_dim
        if self.lambda_dim is not None:
          self.clip_lmbd_dim=int(lambda_dim/self.N)
          self.lmbd_ubd=lambda_ubd
          

  def _get_Q(self, input, n_tv):
      '''
      constructs Q from input
      Q = [[ego x, ego r], [tv x, tv p],...]: np.ndarray ## -> th.Tensor
      '''
      obs = observation_unflatten(input,n_tv = n_tv) #check if it works with batched inputs
      ittc=obs['ttc']
      Q = th.hstack([obs['x0'].reshape(1,-1),th.tensor([obs['ev_route']], device="cuda:0").reshape(1,-1)]) #1st row of Q
      for i, tv_x in enumerate(obs['o0']):
        Q = th.vstack((Q,th.hstack([tv_x.reshape(1,-1), th.tensor([obs['mmpreds'][i]], device="cuda:0").reshape(1,-1) ])))
      return self._graph_encoder(Q, ittc)


  def _graph_encoder(self, Q, ittc):
      '''
      compute ttc encoding as 
      Q_new[i]= Q[i]+ ittc[i]
      '''
      Q_new=Q+th.tensor(th.diag(ittc), device="cuda:0")@th.ones_like(Q, device="cuda:0")
      return self.lift(Q_new)
  
  def _clip(self, state):
      lambda_dv, mu_dv = state[:self.clip_lmbd_dim], state[self.clip_lmbd_dim:]

      #lambda (dual variables for l1 norm) clamping
      lambda_clipped = th.clamp(lambda_dv, 0, self.lmbd_ubd )

      #mu (dual vars for ca constrst) clipping
      mu_clipped = th.clamp(mu_dv, 0)

      return th.concat((lambda_clipped, mu_clipped))
      

  def __call__(self, x):
      '''
      x: th.Tensor
      out: th.Tensor
      '''
      ## Encoder ####
      batch_size=x.shape[0]
      n_tv = int((x.shape[1] - 5) / 4)

      Q=th.stack([self._get_Q(x[i],n_tv) for i in range(batch_size)])
      attn, _ =self.mh_attn(Q,Q,Q)
      x=self.add_norm(Q+attn)
      x=self.add_norm(x+self.pred(x))

      ## Recurrent units

      h_0 = th.zeros_like(x)

      h=h_0

      l1_duals=[]; ca_duals = []

      for _ in range(self.N):
        attn, _=self.mh_attn_dc(x,x,h)
        attn=self.add_norm(x+attn)
        h=self.add_norm(attn+self.pred_d(attn)) #shape: (n_batch, n_tv + 1, embed_dim)
        duals = self._clip(self.project(th.flatten(h,start_dim=1))) #shape: (n_batch, lambda_dim + mu_dim)
        l1_duals.append(duals[:,:self.clip_lmbd_dim])
        ca_duals.append(duals[:,self.clip_lmbd_dim:])

      return th.hstack((th.hstack(l1_duals).flatten(start_dim=1),th.hstack(ca_duals).flatten(start_dim=1)))
  
class R_Tf_binary(nn.Module):
  '''
    Recurrent Transformer arch
  '''
  def __init__(self,input_dim, embed_dim, output_dim, horizon, num_layers,hidden_size,lambda_dim = None, reduced_mode=True, eps = 0.8, lambda_ubd = 1000, pred_mode=["both duals",'tertiary','binary'],device='cuda:0'):
        super(R_Tf_binary ,self).__init__()
        self.pred_mode = pred_mode
        self.eps = eps
        if reduced_mode:
          self.Q_dim = [4,3]   # num_vs x [state, mode]
        else:
          self.Q_dim = [6,3]
        self.lift=nn.Linear(self.Q_dim[1], embed_dim)
        self.norm = nn.BatchNorm2d(3)  # input, key, value are the features
        self.mh_attn=nn.MultiheadAttention(embed_dim, 1)
        self.add_norm=nn.LayerNorm(embed_dim)

        self.num_layers = num_layers
        self.fc_in = nn.Linear(embed_dim, hidden_size)
        self.hidden_layers = []

        self.embed_dim = embed_dim
        self.output_dim=output_dim
        self.device = device
        for i in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size,hidden_size))
            self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.1))

        self.fc_hidden = nn.Sequential(*self.hidden_layers)
        self.fc_out = nn.Linear(hidden_size, embed_dim)
        self.pred = nn.Sequential(self.fc_in,self.fc_hidden,self.fc_out)

        # self.drop= th.nn.Dropout( p = 0.2)


        # Decoder attn
        self.N=horizon # should be SMPC.N-1       

        self.mh_attn_dc = nn.MultiheadAttention(embed_dim, 1)

        
        self.fc_in_d = nn.Linear(embed_dim, hidden_size)
        self.hidden_layers_d = []

        for i in range(num_layers):
            self.hidden_layers_d.append(nn.Linear(hidden_size,hidden_size))
            self.hidden_layers_d.append(nn.LeakyReLU(negative_slope=0.1))

        self.fc_hidden_d = nn.Sequential(*self.hidden_layers)
        self.fc_out_d = nn.Linear(hidden_size, embed_dim)
        self.pred_d = nn.Sequential(self.fc_in_d,self.fc_hidden_d,self.fc_out_d)

        self.drop_dec = th.nn.Dropout( p = 0.1)

        self.project = nn.Linear(embed_dim * self.Q_dim[0], int(output_dim/self.N*3) if self.pred_mode[0]== "l1" and self.pred_mode[1]=='tertiary' else int(output_dim/self.N)) 

        self.rnn_d=nn.GRU(embed_dim, embed_dim*self.Q_dim[0], batch_first=True)
        self.sigmoid_act = nn.Sigmoid()
        self.log_softmax = th.nn.LogSoftmax(dim=-1)
        
        #clip the output
        self.lambda_dim=lambda_dim if self.pred_mode[0]=="both duals" else output_dim
       
        self.clip_lmbd_dim=int(self.lambda_dim/self.N)
        self.lmbd_ubd=lambda_ubd
        if self.pred_mode[0] == "ca":
           self.clip_fn = None
           #Exp. decaying weights throughout the prediction horizon (Assume N_TV = 3)
          #  N_TV = 3
          #  decay_tensor = th.pow(0.5, th.arange(self.N-1)) * 10
          #  w1 = decay_tensor.repeat(int(output_dim/(self.N*N_TV)))
          #  w2 = decay_tensor.repeat(int(output_dim/(self.N*N_TV)))
          #  w3 = decay_tensor.repeat(int(output_dim/(self.N*N_TV)))
          #  self.exp_W  = th.hstack([w1,w2,w3])
          #  self.clip_fn = lambda x: self.exp_W * x
        else:
            if self.pred_mode[1] == 'tertiary':
              self.clip_fn = None
            else:
              self.clip_fn = lambda x: (th.tanh(4*x) + 1) /2 #This is needed for cont. pred 



  def _get_Q(self, input, n_tv):
      '''
      constructs Q from input
      Q = [[ego x, ego r], [tv x, tv p],...]: np.ndarray ## -> th.Tensor
      '''
      obs = observation_unflatten(input,n_tv = n_tv) #check if it works with batched inputs
      ittc=obs['ttc']
      Q = th.hstack([obs['x0'].reshape(1,-1),1e3*th.tensor([obs['ev_route']], device=self.device).reshape(1,-1)]) #1st row of Q
      for i, tv_x in enumerate(obs['o0']):
        Q = th.vstack((Q,th.hstack([tv_x.reshape(1,-1), 1e3*th.tensor([obs['mmpreds'][i]], device=self.device).reshape(1,-1) ])))
      return self._graph_encoder(Q, ittc)


  def _graph_encoder(self, Q, ittc):
      '''
      compute ttc encoding as 
      Q_new[i]= Q[i]+ ittc[i]
      '''
      Q_new=Q+th.tensor(th.diag(ittc), device=self.device)@th.ones_like(Q, device=self.device)
      return self.lift(Q_new)
  
  def _clip(self, state):
      if self.pred_mode[0] == "both duals":
        if self.pred_mode[1] == 'tertiary':
          lambda_dv, mu_dv = state[:,:self.clip_lmbd_dim], state[:,self.clip_lmbd_dim:]
        else:
          lambda_dv, mu_dv = state[:,:self.clip_lmbd_dim], state[:,self.clip_lmbd_dim:]
        return th.concat((self.clip_fn(lambda_dv), mu_dv),dim=1)  
      else:
            return state if self.clip_fn is None else self.clip_fn(state)
      

  def __call__(self, x):
      '''
      x: th.Tensor
      out: th.Tensor
      '''
      ## Encoder ####
      batch_size=x.shape[0]
      n_tv = int((x.shape[1] - 5) / 4)

      Q=th.stack([self._get_Q(x[i],n_tv) for i in range(batch_size)])
      # Q_n = self.norm(th.stack([Q,Q,Q], dim=1))
      # Q = Q_n[:,0,:,:]
      attn, _ =self.mh_attn(Q,Q,Q)
      # attn = self.drop(attn)
      x=self.add_norm(Q+attn)
      x=self.add_norm(x+self.pred(x))

      ## Recurrent units

      h_0 = th.zeros_like(x)

      h=h_0
      if self.pred_mode[0]=="both duals":
        l1_duals=[]; ca_duals = []

        for _ in range(self.N):
          attn, _=self.mh_attn_dc(x,x,h)
          attn = self.drop_dec(attn)
          attn=self.add_norm(x+attn)
          h=self.add_norm(attn+self.pred_d(attn)) #shape: (n_batch, n_tv + 1, embed_dim)
          duals = self._clip(self.project(th.flatten(h,start_dim=1))) #shape: (n_batch, lambda_dim + mu_dim)
          x_o, h_o = self.rnn_d(x, th.stack([th.flatten(h,start_dim=1)]))
          # h = h_o[0,:,:].view(batch_size, n_tv+1, -1)
          # x, h = self.rnn_d(x, h)
          h = h_o[0,:,:].view(batch_size, n_tv+1, -1)
          x = x_o[:,:,:self.embed_dim]

          l1_duals.append(duals[:,:self.clip_lmbd_dim])
          ca_duals.append(duals[:,self.clip_lmbd_dim:])

        return th.hstack((th.hstack(l1_duals).flatten(start_dim=1),th.hstack(ca_duals).flatten(start_dim=1)))
      else:
        duals = []
        for k in range(self.N):
          attn, _=self.mh_attn_dc(x,x,h)
          attn = self.drop_dec(attn)
          attn=self.add_norm(x+attn)
          h=self.add_norm(attn+self.pred_d(attn)) #shape: (n_batch, n_tv + 1, embed_dim)
          dual = self._clip(self.project(th.flatten(h,start_dim=1))) #shape: (n_batch, lambda_dim + mu_dim)

          if self.pred_mode[0] == 'l1' and self.pred_mode[1] == 'tertiary':
             temp = dual.view(batch_size,int(self.output_dim/self.N),3)   
             dual = self.log_softmax(temp)  #shape: (N_batch, l1_dim/N, 3)       
          x_o, h_o = self.rnn_d(x, th.stack([th.flatten(h,start_dim=1)]))
          # h = h_o[0,:,:].view(batch_size, n_tv+1, -1)
          # x, h = self.rnn_d(x, h)
          h = h_o[0,:,:].view(batch_size, n_tv+1, -1)
          x = x_o[:,:,:self.embed_dim]

          if self.pred_mode[0] == 'ca':
            duals.append(dual[:,:self.clip_lmbd_dim]*(1+self.eps**k))
          else:
            duals.append(dual[:,:self.clip_lmbd_dim])
          
        return th.hstack(duals) if self.pred_mode[0] == 'l1' and self.pred_mode[1] == 'tertiary' else th.hstack(duals).flatten(start_dim=1)
