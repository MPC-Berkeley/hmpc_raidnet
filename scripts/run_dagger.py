import gymnasium as gym
from infrastructure.policies import R_Tf_binary as R_Tf #Import desired policy
import yaml
from infrastructure.utils import observation_flatten
import os
import time
from infrastructure.logger import Logger
import torch as th
import warnings
from scripts.dagger import dagger
warnings.filterwarnings( 'ignore' )
import argparse

def main(args):

    #Import training configuration parameters
    with open(args.config_dir, 'r') as file:
        config = yaml.safe_load(file)

    #Create and register the custom gym environment
    gym.register(id='traffic_env-v0',entry_point='infrastructure.sim_gym:TrafficEnv')
    env = gym.make('traffic_env-v0',reduced_mode=config['reduced_mode'],N=config['N'])
    reduced_mode = config['reduced_mode']; N = config['N']
    print(f'Initializing traffic_env-v0 with reduced_mode: {reduced_mode}, N: {N}')

    obs, _ = env.reset()
    observation_dim = len(observation_flatten(obs).flatten())
    ca_num = len(env.smpc.mode_map)*(env.smpc.N-1)*env.smpc.N_TV
    l1_num = sum(env.smpc.N_modes)*(env.smpc.N-1)*2
    action_dim = ca_num + l1_num
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]

    lambda_dim, lambda_ubd = l1_num, env.smpc.l1_lmbd #Define l1 dual var dim and ubd for the neural network model
    policy=R_Tf(observation_dim, 2*observation_dim, ca_num, env.smpc.N-1, num_layers, hidden_size, lambda_dim=lambda_dim,lambda_ubd = lambda_ubd,pred_mode=['ca','tertiary','binary'])
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy.to(device)
 
    policy_type="RAIDNET"

    #Collect Data
    collect_data = False

    #Initiate logger
    logdir = (
        policy_type
        + "_"
        + "dagger_"
        + 'TrafficEnv'
        + "_"
        + "N" + str(env.smpc.N)
        + "_N_TV" + str(3 if env.reduced_mode else 5)
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(config['root_dir'] + config['training_log_path'], logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(logdir)

    #Load a pre-trained model
    load_model = True

    if load_model:
        dagger_learner = dagger(env,policy=policy,collect_data=collect_data,save_expert_data=True,config=config,logger=logger,model_dir=config["model_dir"],policy_type=policy_type)
    else:
        dagger_learner = dagger(env,policy=policy,collect_data=collect_data,save_expert_data=True,logger=logger,config=config,policy_type=policy_type)

    if collect_data:
        dagger_learner.collect_exp_data()

    bc_learner = dagger_learner.learn(initial_train=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir',type=str, required=True,help='Evaluation Parameter Configuration File is not provided')
    args = parser.parse_args()
    main(args)