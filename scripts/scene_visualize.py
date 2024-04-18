#!/bin/bash
import gymnasium as gym
from infrastructure.policies import R_Tf_binary #Import desired policy
import yaml
from infrastructure.utils import observation_flatten, sample_trajectory
import torch as th
import warnings
warnings.filterwarnings( 'ignore' )
import os
import pdb
import matplotlib.pyplot as plt
from celluloid import Camera

def visualize(ind):
    fig, ax= plt.subplots()
    camera = Camera(fig)
    env.Sim.draw_intersection(ax,ind)
    artists = camera.snap()
    for element in artists:
        ax.add_artist(element)

    # Show the plot
    plt.show()

def load(policy, model_dir):
    save_file = None
    if os.path.exists(model_dir):
        save_file = model_dir
    if save_file is not None:
        checkpoint = th.load(save_file)
        print('Model loaded: {}'.format(save_file))
        policy.load_state_dict(checkpoint['model_state_dict'])
        # self.bc_learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("ERROR: MODEL NOT LOADED")

#Import training configuration parameters
# with open('data/configs/params.yaml', 'r') as file:
#     config = yaml.safe_load(file)
with open('data/configs/params_N14.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Create and register the custom gym environment
gym.register(id='traffic_env-v0',entry_point='infrastructure.sim_gym:TrafficEnv')
solver = config['solver']

env = gym.make('traffic_env-v0',reduced_mode=config['reduced_mode'],N=config['N'],env_mode=2,solver = solver)
reduced_mode = config['reduced_mode']; N = config['N']
print(f'Initializing traffic_env-v0 with reduced_mode: {reduced_mode}, N: {N}')

# obs, _ = env.reset()
observation_dim = 17
ca_num = len(env.smpc.mode_map)*(env.smpc.N-1)*env.smpc.N_TV
l1_num = sum(env.smpc.N_modes)*(env.smpc.N-1)*2
action_dim = ca_num + l1_num
num_layers = config["num_layers"]
hidden_size = config["hidden_size"]

lambda_dim, lambda_ubd = l1_num, env.smpc.l1_lmbd #Define l1 dual var dim and ubd for the neural network model

#Intialize and Load policy
if config['joint_dual_pred']:
    policy=R_Tf_binary(observation_dim, 2*observation_dim, action_dim, env.smpc.N-1, num_layers, hidden_size, lambda_dim=lambda_dim,lambda_ubd = lambda_ubd)
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy.to(device)
    policy_type="RAIDNET"
    load(policy,config['model_dir'])
else:
    l1_policy = R_Tf_binary(observation_dim, 2*observation_dim, lambda_dim, env.smpc.N-1, num_layers//2, hidden_size//2, lambda_dim=lambda_dim,lambda_ubd = lambda_ubd,pred_mode=['l1','tertiary','binary'])
    ca_policy = R_Tf_binary(observation_dim, 2*observation_dim, ca_num, env.smpc.N-1, num_layers//2, hidden_size//2, lambda_dim=lambda_dim,lambda_ubd = lambda_ubd,pred_mode=['ca','tertiary','binary'])
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    l1_policy.to(device)
    ca_policy.to(device)
    policy = [l1_policy, ca_policy]
    load(l1_policy,config['l1_model_dir'])
    load(ca_policy,config['ca_model_dir'])
    policy_type="RAIDNET"

#Define the save path
save_dir = config['eval_save_dir']
if config['joint_dual_pred']:
    model_name = config['model_dir'].split('/')[-1].split('.')[0]
else:
    model_name = config['l1_model_dir'].split('/')[-1].split('.')[0]
tertiary_l1 = True


df = None
env.Sim.viz_preds = False
path_policy =  sample_trajectory(env,policy=policy,max_path_length=config["eval_max_path_length"],use_cuda=True,render=False, expert=False,binary_pred=config['binary_pred'],tertiary_l1=tertiary_l1,normalize_obs=config['normalize_obs'])
pdb.set_trace()
visualize(10)

