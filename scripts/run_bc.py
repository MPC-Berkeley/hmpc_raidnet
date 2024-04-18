import gymnasium as gym
from infrastructure.policies import R_Tf_binary as R_Tf
import numpy as np
import yaml
from infrastructure.utils import observation_flatten
from infrastructure.replay_buffer import ReplayBuffer
import pdb
import os
import time
import pickle
from infrastructure.logger import Logger
import torch as th
from scripts.bc import BC
import warnings
warnings.filterwarnings( 'ignore' )
import argparse

def Train_BC(env,policy,policy_type,device,config,pretrain=False,checkpoint=[]):
    #Initiate logger
    logdir = (
        policy_type
        + "_"
        + 'TrafficEnv'
        + "_"
        + "N" + str(env.smpc.N)
        + "_N_TV" + str(3 if env.reduced_mode else 5)
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(config['root_dir'] + config['training_log_path'], logdir)
    if not (os.path.exists(logdir)):
        print(logdir)
        os.makedirs(logdir)
    logger = Logger(logdir)

    #Run BC
    normalize = True
    normalize_obs = True
    N_EPOCHS = config["N_EPOCHS"]
    optimizer = config["optimizer"]
    lr = config["lr"]

    #Load expert trajectory dataset
    with open(config['expert_trajectory_data'],'rb') as file:
        paths = pickle.load(file)
    replay_buffer = ReplayBuffer(config['max_replay_buffer_size'])
    replay_buffer.obs = paths["observation"]; replay_buffer.acs = paths["action"]; replay_buffer.terminals = paths["terminal"]; replay_buffer.next_obs = paths["next_observation"]; replay_buffer.rews = paths['reward'] 
    replay_buffer.dual_classes = paths["dual_classes"]
    replay_buffer.set_weights()

    #Initialize BC algorithm
    bc_learner = BC(policy=policy,optimizer=optimizer,optim_lr=lr, observation_space= env.observation_space, action_space=env.action_space, demonstrations=replay_buffer, rng = np.random.default_rng(0),device = device, batch_size=config['training_batch_size'],logger=logger,normalize=normalize,config=config, normalize_obs=normalize_obs, l1_dual_dim= [env.smpc.N-1, env.smpc.N_modes, env.smpc.N_TV],ca_dual_dim= [env.smpc.N-1, len(env.smpc.mode_map), env.smpc.N_TV])
    
    if pretrain:
        bc_learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print('TRAINING STARTED'.center(80,'*'))
    training_log = bc_learner.train(n_epochs=N_EPOCHS, n_batches=config["training_batch_size"], binary_pred=True)

    #Save the policy parameters
    bc_learner.save(config['model_save_dir'],config=config)

    with open(config['training_log_path'] + 'training_log.pkl', 'wb') as handle:
        pickle.dump(training_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return bc_learner

def main(args):
    #Import training configuration parameters
    with open(args.config_dir, 'r') as file:
        config = yaml.safe_load(file)

    #Create and register the custom gym environment
    gym.register(id='traffic_env-v0',entry_point='infrastructure.sim_gym:TrafficEnv')
    env = gym.make('traffic_env-v0',reduced_mode=config['reduced_mode'],N=config['N'],env_mode=0)
    reduced_mode = config['reduced_mode']; N = config['N']
    print(f'Initializing traffic_env-v0 with reduced_mode: {reduced_mode}, N: {N}')

    obs, _ = env.reset()
    observation_dim = len(observation_flatten(obs).flatten())
    ca_num = len(env.smpc.mode_map)*(env.smpc.N-1)*env.smpc.N_TV
    l1_num = sum(env.smpc.N_modes)*(env.smpc.N-1)*2
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    device = th.device("cuda:0" if th.cuda.is_available() else 'cpu')

    #Initialize the NN policy
    policy=R_Tf(observation_dim, 2*observation_dim, ca_num, env.smpc.N-1, num_layers//2, hidden_size//2, pred_mode=['ca','tertiary','binary'])
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy.to(device)

    policy_type="RAIDNET"
    pretrain = config['pretrain']
    checkpoint = []

    if pretrain:
        save_file = None
        if os.path.exists(config['root_dir']+config['model_dir']):
            save_file = config['root_dir']+config['model_dir']
        if save_file is not None:
            checkpoint = th.load(save_file)
            print('Model loaded: {}'.format(save_file))
            policy.load_state_dict(checkpoint['model_state_dict'])
        
        else:
            print('Can not find the model for {}'.format(config['root_dir']+config['model_dir']))
    
    bc_learner = Train_BC(env,policy, policy_type,device=device, config=config,pretrain=pretrain,checkpoint=th.load(config['model_dir'])) #Run a single iteration of BC algorithm

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir',type=str, required=True,help='Evaluation Parameter Configuration File is not provided')
    args = parser.parse_args()
    main(args)
