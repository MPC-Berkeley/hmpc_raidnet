from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
import pdb
import gymnasium as gym
import numpy as np
import torch as th
from infrastructure.utils import to_tensor_var, weighted_MSEloss, obs_normalize, unflatten_duals, flatten
# from stable_baselines3.common import policies, utils, vec_env
from datetime import datetime

class BC():
    def __init__(self,
                 policy: None,
                 device: Union[str, th.device],
                 optimizer: 'Adam',
                 optim_lr: 0.001,
                observation_space: gym.Space,
                action_space: gym.Space,
                rng: np.random.Generator,
                demonstrations: None,
                logger: None,
                normalize: False,
                normalize_obs: False,
                config: None,
                ca_dual_dim: None,
                l1_dual_dim: None,
                batch_size: int = 32,
                dagger_mode: bool = False,
                ismlp: bool = False,):
        self.l1_dual_dim = l1_dual_dim
        self.policy = policy
        self.device = device
        self.logger = logger
        self.ca_dual_dim = ca_dual_dim
        self.normalize = normalize
        self.normalize_obs = normalize_obs
        self.file_path = None
        self.demonstrations = demonstrations
        self.dagger_mode = dagger_mode
        self.ismlp = ismlp

        #Weighted sampling
        weights = th.DoubleTensor(self.demonstrations.dataset_weights)
        w_sampler = th.utils.data.sampler.WeightedRandomSampler(weights, self.demonstrations.max_size)
        self.train_loader = th.utils.data.DataLoader(self.demonstrations, batch_size=batch_size,sampler = w_sampler, pin_memory=True) 

        #Setting up the optimizers for the policies
        if optimizer=='Adam':
                self.optimizer = th.optim.AdamW(self.policy.parameters(),  lr=optim_lr)
        else:
            self.optimizer = th.optim.RMSprop(self.policy.parameters(),  lr=optim_lr)
        # self.lr_sched = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 =10, T_mult = 2, eta_min = optim_lr*0.1, last_epoch=-1 )
        
        if th.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False 

        self.set_demonstrations(demonstrations)
        self.action_space = action_space
        self.observation_space = observation_space

        self.rng = rng
        self.config = config


        self.bce_loss = th.nn.BCEWithLogitsLoss(pos_weight=4*th.ones(self.policy.lambda_dim, device=th.device('cuda')))

    def set_demonstrations(self, dataset):
        self.demonstrations = dataset
       
    def update_policy(self, new_policy):
        self.policy = new_policy

    def update_statistics(self, acs, binary = True):
        ca_duals = acs[:, -self.policy.lambda_dim:] 
        acs[:, -self.policy.lambda_dim:] = th.Tensor(ca_duals > 1e-3).float()
        self.ac_std = th.ones(int(acs.shape[1]), device=th.device('cuda'))
    
    def train(
        self,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        binary_pred = False
    ):
        training_log = {}
        self.training_loss = []
        self.epochs = [i for i in range(n_epochs)]

        for itr in range(n_epochs):
            logs = {}
            #Get Training batch
            ob_batch, ac_batch, dual_class_batch = next(iter(self.train_loader)) #for weighted sampling
            
            if self.normalize_obs:
                ob_batch = obs_normalize(ob_batch)

            #Policy Gradient Descent 
            #Compute the loss
            if self.normalize:
                self.update_statistics(ac_batch, binary_pred)
                expert_ac_batch = to_tensor_var([np.fromiter(flatten(unflatten_duals(ac_batch[k,:][None],self.l1_dual_dim, self.ca_dual_dim, data2tar=True)),float) for k in range(n_batches)], use_cuda=self.use_cuda) / (self.ac_std )
                
            else:
                expert_ac_batch = to_tensor_var([np.fromiter(flatten(unflatten_duals(ac_batch[k,:][None],self.l1_dual_dim, self.ca_dual_dim, data2tar=True)),float) for k in range(n_batches)], use_cuda=self.use_cuda)

            self.loss = []
            self.optimizer.zero_grad()

            correct = th.sum(th.sigmoid(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))).round()==expert_ac_batch[:,-self.policy.lambda_dim:])
            sum_pred = th.sum(th.sigmoid(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))).round()).item()
            sum_tar  = th.sum(expert_ac_batch[:,-self.policy.lambda_dim:]).item()
            training_ca_acc = correct.item()/(n_batches*(self.policy.output_dim))
            print("Correct CA: ",correct.item(), " out of ", n_batches*(self.policy.output_dim),training_ca_acc*100,"% acc")
            print("Ones in pred CA: ", sum_pred, " Ones in target CA:",sum_tar)
            self.loss = self.bce_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda)),expert_ac_batch[:,-self.policy.lambda_dim:])

            self.loss.backward()
            self.optimizer.step()
            if hasattr(self,'lr_sched'):
                self.lr_sched.step()
            loss_value = self.loss.detach().cpu().numpy()
            self.training_loss.append(loss_value) #for batches

            #Logging
            if self.logger:
                logs.update({'Epochs':itr, 'Training_Loss':self.loss.detach().cpu().numpy(), 'Training_EnvStepsSofar': n_batches * itr})

                if not self.dagger_mode:
                    for key, value in logs.items():
                        print("{} : {}".format(key, value))
                        self.logger.log_scalar(value, key,itr)
                    print("Done logging...\n\n")
                    print('-'.center(80,'-'))
                    self.logger.flush()

            #OPTIONAL (save):
            if itr % self.config['model_save_period']== 0 and itr != 0:
                self.save(self.config['model_save_dir'],config=self.config,iter=itr)
            training_log.update({'training_batch_size': n_batches,'ca_training_loss': np.concatenate([self.training_loss]), 'Epochs': self.epochs,'training_ca_acc': training_ca_acc,'Training_EnvStepsSofar': n_batches * itr,})
        return training_log 

    def save(self, model_save_dir, config,iter=None):
        
        if not self.file_path:
            now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
            self.file_path = model_save_dir + 'RAIDNET_BC_' + now + '_'
        
        th.save({'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),'config': config},
                self.file_path + str(iter) + 'epoch'+ '.pt')
        
        
