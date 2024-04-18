"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""

from collections import OrderedDict
import numpy as np
import time

from infrastructure import pytorch_util as ptu

from torch.autograd import Variable
from collections.abc import Iterable
import numpy as np
import torch as th
import pdb
import time
import os
import copy

def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    dual_classes = np.concatenate([path["dual_classes"] for path in paths])
    return observations, actions, rewards, next_observations, terminals, dual_classes

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x #creates a generator object

def unflatten_duals(x,l1_dual_dim,ca_dual_dim,data2tar = False):

    l1_num=sum(l1_dual_dim[1])*(l1_dual_dim[0])*2       

    l1_dual_arr = x[0,:l1_num].flatten().tolist() 
    ca_dual_arr = x[0,l1_num:].flatten().tolist()

    if data2tar:
        l1_dual = [[[ []  for j in range(l1_dual_dim[1][k])] for k in range(l1_dual_dim[2])] for t in range(l1_dual_dim[0])]
        ca_dual = [[[ []  for j in range(ca_dual_dim[1])] for k in range(ca_dual_dim[2])]  for t in range(ca_dual_dim[0])] 

        step=0
        for t in range(l1_dual_dim[0]):
            for k in range(l1_dual_dim[2]):
                for j in range(l1_dual_dim[1][k]):

                    l1_dual[t][k][j]+=[l1_dual_arr[step:step+2]]
                    step+=2
        step=0
        for t in range(ca_dual_dim[0]):
            for k in range(ca_dual_dim[2]):
                for j in range(ca_dual_dim[1]):

                    ca_dual[t][k][j]+=list(ca_dual_arr[step:step+1])
                    step+=1     
    else:
        l1_dual = [[[ [] for t in range(l1_dual_dim[0])] for j in range(l1_dual_dim[1][k])] for k in range(l1_dual_dim[2])]
        ca_dual = [[[ [] for t in range(ca_dual_dim[0])] for j in range(ca_dual_dim[1])] for k in range(ca_dual_dim[2])]
    

        step=0
        for t in range(l1_dual_dim[0]):
            for k in range(l1_dual_dim[2]):
                for j in range(l1_dual_dim[1][k]):

                    l1_dual[k][j][t]+=[l1_dual_arr[step:step+2]]
                    step+=2
        step=0
        for t in range(ca_dual_dim[0]):
            for k in range(ca_dual_dim[2]):
                for j in range(ca_dual_dim[1]):

                    ca_dual[k][j][t]+=list(ca_dual_arr[step:step+1])
                    step+=1

    return l1_dual, ca_dual

def obs_normalize(obs, reduced_mode =True):
    '''
    Assume reduced_mode = True
    obs #shape (N, 17)
    "mmpreds" : MultiDiscrete([4,3,5])
    '''
    obs_norm = copy.deepcopy(obs)
    
    def clip(input,min,max):
        if isinstance(input,th.Tensor):
            return th.clip(input,min, max)
        else:
            return NotImplementedError
        
    if reduced_mode:
        #Ego normalization
        obs_norm[:,0] = (obs[:,0] / 110)
        v_max = 10; v_min = -1
        obs_norm[:,1] = (clip(obs[:,1],v_min, v_max) - v_min)/(v_max - v_min) #min-max normalization
        a_max = 2; a_min = -5
        obs_norm[:,2] = (clip(obs[:,2], a_min, a_max) - a_min) / (a_max - a_min) #min-max normalization
        obs_norm[:,3] /= 2 #ego route: Discrete(2)

        #ittc normalization
        obs_norm[:,4:4+4] = (clip(obs_norm[:,4:4+4], 0.05, 10) - 0.05)/ (10 - 0.05)

        #o0 normalization
        obs_norm[:,8] /= th.where(obs[:,8] == -15., th.tensor(15.), th.tensor(110))
        obs_norm[:,10] /= th.where(obs[:,10] == -15., th.tensor(15.), th.tensor(110))
        obs_norm[:,12] /= th.where(obs[:,12] == -15., th.tensor(15.), th.tensor(110))

        obs_norm[:,9] = (clip(obs_norm[:,9],v_min, v_max) - v_min)/(v_max - v_min)
        obs_norm[:,11] = (clip(obs_norm[:,9],v_min, v_max) - v_min)/(v_max - v_min)
        obs_norm[:,13] = (clip(obs_norm[:,9],v_min, v_max) - v_min)/(v_max - v_min)

        #mmpreds normalization
        obs_norm[:,14] /= 4 - 1
        obs_norm[:,15] /= 3 - 1
        obs_norm[:,16] /= 5 - 1

    return obs_norm
    

def observation_flatten(obs, use_ttc=True):
    if use_ttc:
        return np.concatenate([ obs['x0'],np.array([obs['u_prev'],obs['ev_route']]), obs['ttc'], np.stack(obs['o0'],axis=0).flatten(),obs['mmpreds']]).astype('float32')
    else:
        return np.concatenate([ obs['x0'],np.array([obs['u_prev'],obs['ev_route']]), np.stack(obs['o0'],axis=0).flatten(),obs['mmpreds']]).astype('float32')

def observation_unflatten(obs_flat, n_tv, use_ttc=True):
    if len(obs_flat.shape) > 1: #If batch
        o0_arr = obs_flat[(1+n_tv)*int(use_ttc) + 4 : (1+n_tv)*int(use_ttc) + 4 + n_tv * 2].reshape(-1,2)
        obs_dict={'x0': obs_flat[0:2], 'u_prev': obs_flat[2], 'ev_route': int(obs_flat[3]), 'o0': [o0_arr[ind,:] for ind in range(o0_arr.shape[0])], 'mmpreds':obs_flat[(1+n_tv)*int(use_ttc)+4 + n_tv * 2:] }
        if use_ttc:
            obs_dict.update({'ttc':obs_flat[4:4+1+n_tv] })
        return obs_dict
    else:    
        o0_arr = obs_flat[(1+n_tv)*int(use_ttc)+4:(1+n_tv)*int(use_ttc)+4 + n_tv * 2].reshape(-1,2)
        obs_dict={'x0': obs_flat[0:2], 'u_prev': obs_flat[2], 'ev_route': int(obs_flat[3]), 'o0': [o0_arr[ind,:] for ind in range(o0_arr.shape[0])], 'mmpreds':obs_flat[(1+n_tv)*int(use_ttc)+4 + n_tv * 2:] }
        if use_ttc:
            obs_dict.update({'ttc':obs_flat[4:4+1+n_tv] })
        return obs_dict
        
def sample_trajectory(env, policy=None, max_path_length=100, use_cuda = False,seed=None,render=False,ani_save_dir=None,expert=True, binary_pred= True,tertiary_l1 = False, normalize_obs=False,dagger_mode=False): 
    """Sample a rollout in the environment from a policy."""
    print('Sampling a trajectory...')
    rollout_done = False
    ob, info = env.reset(seed=seed)
    obs, acs, rewards, next_obs, terminals, solve_times, infeas, collisions, vars_kept, const_kept, NN_query_times, dual_classes= [], [], [], [], [], [], [], [], [], [], [], []
    t_wall_sums, t_proc_sums = [], []
    steps = 0
    only_ca_pred = True
    

    while steps <= max_path_length and not rollout_done:
        # print(f"Steps {steps}".center(80,'-'))
        if policy is None:
            new_ob, reward, done, _, infos = env.step(action=None)
            NN_query_time = 0
        else:
            st = time.time()
            l1_pred = th.zeros((1,sum(env.smpc.N_modes)*(env.smpc.N-1)*2)).to(device="cuda" if use_cuda else "cpu") #Dummy l1 duals required for downstream smpcfr.py
            ca_pred = th.sigmoid(policy(obs_normalize(to_tensor_var(observation_flatten(ob), use_cuda=use_cuda)[None]) if normalize_obs else to_tensor_var(observation_flatten(ob), use_cuda=use_cuda)[None])).round()
            NN_query_time = time.time() - st
            action = th.hstack((l1_pred,ca_pred))
            
            l1_dual, ca_dual = unflatten_duals(action.detach().cpu().numpy(), [env.smpc.N-1, env.smpc.N_modes, env.smpc.N_TV], [env.smpc.N-1, len(env.smpc.mode_map), env.smpc.N_TV] )
            action = [l1_dual, ca_dual]
            new_ob, reward, done, _, infos = env.step(action=action)        
            # print("Step taken ",new_ob["x0"] )

        steps += 1
        rollout_done = done or infos['discard']
        if rollout_done:
            if infos['infeas']:
                infeas.append(infos['infeas'])
            
        if not infos['infeas']:
            l1_duals = np.fromiter(flatten(infos["l1_duals"]),float)
            ca_duals = np.fromiter(flatten(infos["ca_duals"]),float)
            expert_action = np.concatenate((l1_duals,ca_duals))
            action = expert_action
            
            obs.append(observation_flatten(ob))
            acs.append(action)
            rewards.append(reward)
            next_obs.append(observation_flatten(new_ob))
            terminals.append(rollout_done)
            solve_times.append(infos['solve_time'])
            NN_query_times.append(NN_query_time)
            infeas.append(infos['infeas'])
            collisions.append(infos['discard'])
            dual_classes.append(infos["dual_class"])
            if 'vars_kept' in infos.keys():
                vars_kept.append(infos['vars_kept'])
                const_kept.append(infos['const_kept'])

            if env.env_mode == 2:
                t_wall_sums.append(infos['t_wall_sum'])
                t_proc_sums.append(infos['t_proc_sum'])
            else:
                #Append placeholders
                t_wall_sums.append(-1)
                t_proc_sums.append(-1)
        elif infos['infeas'] and not dagger_mode:
            infeas.append(infos['infeas'])
            solve_times.append(None)
            NN_query_times.append(None)
            collisions.append(infos['discard'])
            dual_classes.append(None)
            t_wall_sums.append(None)
            t_proc_sums.append(None)
            vars_kept.append(None)
            const_kept.append(None)            
        ob = new_ob
    print(f'Steps: {steps}')
    if not infos['discard'] or dagger_mode:
        path = {'observation':obs,'reward': np.array(rewards, dtype=np.float32), 'action': np.array(acs, dtype=np.float32),'next_observation': next_obs, "terminal": np.array(terminals, dtype=np.float32), "infeas": infeas, "solve_time":solve_times, 'collision': collisions, 'vars_kept': vars_kept, 'const_kept':const_kept, 'NN_query_time': NN_query_times, "dual_classes":dual_classes, 't_wall_sum': t_wall_sums, 't_proc_sum':t_proc_sums}     #state and expert action  
    else:
        path = None

    if render:
        animation = env.render()
        if expert:
            name = 'expert'
        else:
            name = 'HMPC'
        if os.path.isdir(ani_save_dir):
            pass
        else:
            os.mkdir(ani_save_dir)
        animation.save(ani_save_dir + 'eval_' + name +'.mp4')
    return path

def get_pathlength(path):
    return len(path["reward"])

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, use_cuda = False,seed=None,tertiary_l1 = False,normalize_obs=False,dagger_mode=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length,use_cuda=use_cuda,seed=seed,tertiary_l1 = tertiary_l1,normalize_obs=normalize_obs,dagger_mode=dagger_mode)
        if path is not None: #if not discard
            paths.append(path)
            timesteps_this_batch += get_pathlength(path)
        else:
            seed += 1
        

    return paths, timesteps_this_batch
    
def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    
class weighted_MSEloss(th.nn.Module):
    '''
    weighted mse loss for prioritising important l1_duals in target
    '''
    def __init__(self, lmbd_bnd):
        super().__init__()
        self.lmbd_bnd = lmbd_bnd

    def __call__(self, input, target):
        '''
        state: th.Tensor
        out: th.Tensor
        '''
        lmbd_bool = ((target < 1e-8) | (target > self.lmbd_bnd - 1e-8)).float()*49 + th.ones_like(target)
        out = th.nn.functional.mse_loss(lmbd_bool*input, lmbd_bool*target)

        return out

