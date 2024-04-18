import gymnasium as gym
from infrastructure.policies import R_Tf_binary #Import desired policy
import yaml
from infrastructure.utils import observation_flatten, sample_trajectory
import torch as th
import warnings
warnings.filterwarnings( 'ignore' )
import os
import pandas
import numpy as np
import random
import argparse

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

def main(args):
    #Import training configuration parameters
    with open(args.config_dir, 'r') as file:
        config = yaml.safe_load(file)
    N_EVAL = config['N_eval']
    solver = config['solver']

    #Create and register the custom gym environment
    gym.register(id='traffic_env-v0',entry_point='infrastructure.sim_gym:TrafficEnv')
    env = gym.make('traffic_env-v0',reduced_mode=config['reduced_mode'],N=config['N'],env_mode=2,solver = solver)
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

    #Intialize and Load policy
    policy=R_Tf_binary(observation_dim, 2*observation_dim, ca_num, env.smpc.N-1, num_layers//2, hidden_size//2, lambda_dim=lambda_dim,lambda_ubd = lambda_ubd,pred_mode=['ca','tertiary','binary'])
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy.to(device)
    load(policy,config['root_dir']+config['model_dir'])


    #Define the save path
    save_dir = config['root_dir'] + config['eval_save_dir']
    print(f'Recording evaluation metrics to {save_dir}')
    model_name = (config['root_dir'] + config['model_dir']).split('/')[-1].split('.')[0]
    

    '''
    Evaluation
    '''
    tertiary_l1 = True
    gen_seed = 2024
    random.seed(gen_seed)
    seeds = random.choices(range(1000),k=N_EVAL)

    for i in range(N_EVAL):
        df = None
        print(f'Evaluation {i+1} out of {N_EVAL}'.center(80,'-'))
        #Expert rollout
        print('Expert Rollout Started...')
        print(f'seed: {seeds[i]}')
        path_exp = sample_trajectory(env,policy=None,max_path_length=config["eval_max_path_length"],use_cuda=True,seed=seeds[i],render=True,ani_save_dir=save_dir + 'eval_' + str(i) + '_'+ model_name + '_'+ solver + '_' + str(seeds[i]) + '/',expert=True,binary_pred=config['binary_pred'],tertiary_l1=tertiary_l1)
        if path_exp:
            #RAID-Net + MPC rollout
            print('RAID-Net + MPC Rollout Started...')
            path_policy =  sample_trajectory(env,policy=policy,max_path_length=config["eval_max_path_length"],use_cuda=True,seed=seeds[i],render=True,ani_save_dir=save_dir + 'eval_' + str(i) + '_'+ model_name + '_'+ solver + '_' + str(seeds[i]) + '/', expert=False,binary_pred=config['binary_pred'],tertiary_l1=tertiary_l1,normalize_obs=config['normalize_obs'])
            if path_exp is not None and path_policy is not None:
                #Log infeasibility, solve_time, and collision
                if df is not None:
                    length = max(len(path_exp['collision']),len(path_policy['collision']))
                    df = pandas.concat([df,pandas.DataFrame({'traj': [i for j in range(length)],'collision_exp':np.concatenate((path_exp['collision'],[np.nan for _ in range(length-len(path_exp['collision']))])),'solve_time_exp_arr':np.concatenate((path_exp['solve_time'],[np.nan for _ in range(length-len(path_exp['solve_time']))])),'infeas_exp_arr':np.concatenate((path_exp['infeas'],[np.nan for _ in range(length-len(path_exp['infeas']))])),'collision_policy':np.concatenate((path_policy['collision'],[np.nan for _ in range(length-len(path_policy['collision']))])),'solve_time_policy_arr':np.concatenate((path_policy['solve_time'],[np.nan for _ in range(length-len(path_policy['solve_time']))])),'infeas_policy_arr':np.concatenate((path_policy['infeas'],[np.nan for _ in range(length-len(path_policy['infeas']))])), 'vars_kept':np.concatenate((path_policy['vars_kept'],[np.nan for _ in range(length-len(path_policy['vars_kept']))])), 'const_kept':np.concatenate((path_policy['const_kept'],[np.nan for _ in range(length-len(path_policy['const_kept']))])),'NN_query_time':np.concatenate((path_policy['NN_query_time'],[np.nan for _ in range(length-len(path_policy['NN_query_time']))])),'pol_t_wall_sum':np.concatenate((path_policy['t_wall_sum'],[np.nan for _ in range(length-len(path_policy['t_wall_sum']))])),'pol_t_proc_sum':np.concatenate((path_policy['t_proc_sum'],[np.nan for _ in range(length-len(path_policy['t_proc_sum']))]))})])
                else:
                    length = max(len(path_exp['collision']),len(path_policy['collision']))
                    df = pandas.DataFrame({'traj': [i for j in range(length)],'collision_exp':np.concatenate((path_exp['collision'],[np.nan for _ in range(length-len(path_exp['collision']))])),'solve_time_exp_arr':np.concatenate((path_exp['solve_time'],[np.nan for _ in range(length-len(path_exp['solve_time']))])),'infeas_exp_arr':np.concatenate((path_exp['infeas'],[np.nan for _ in range(length-len(path_exp['infeas']))])),'collision_policy':np.concatenate((path_policy['collision'],[np.nan for _ in range(length-len(path_policy['collision']))])),'solve_time_policy_arr':np.concatenate((path_policy['solve_time'],[np.nan for _ in range(length-len(path_policy['solve_time']))])),'infeas_policy_arr':np.concatenate((path_policy['infeas'],[np.nan for _ in range(length-len(path_policy['infeas']))])), 'vars_kept':np.concatenate((path_policy['vars_kept'],[np.nan for _ in range(length-len(path_policy['vars_kept']))])), 'const_kept':np.concatenate((path_policy['const_kept'],[np.nan for _ in range(length-len(path_policy['const_kept']))])),'NN_query_time':np.concatenate((path_policy['NN_query_time'],[np.nan for _ in range(length-len(path_policy['NN_query_time']))])),'pol_t_wall_sum':np.concatenate((path_policy['t_wall_sum'],[np.nan for _ in range(length-len(path_policy['t_wall_sum']))])),'pol_t_proc_sum':np.concatenate((path_policy['t_proc_sum'],[np.nan for _ in range(length-len(path_policy['t_proc_sum']))]))})
                
                df.to_csv(save_dir + 'eval_' + str(i) + '_'+ model_name +'_' + solver + '_' + str(seeds[i]) + '/' + 'eval_' + str(i) + '_'+ model_name + '_'+ solver + '.csv',index=False)
            else:
                print(f'Collision occurred in RAID-Net + MPC Rollout. Discarding this run...')
                f = open(save_dir + 'eval_' + str(i) + '_'+ model_name +'_' + solver + '_' + str(seeds[i]) +'/' + "collision.txt", "w")
                f.write("RAID-Net")
                f.close()
        else:
            print(f'Collision occurred in the Expert Rollout. Discarding this run...')
            f = open(save_dir + 'eval_' + str(i) + '_'+ model_name +'_' + solver + '_' + str(seeds[i]) +'/' + "collision.txt", "w")
            f.write("expert")
            f.close()

    config_filename = save_dir + 'eval_' + str(i) + '_'+ model_name +'_' + solver + '_' + str(seeds[i]) +'/' +'params.yaml'
    with open(config_filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    print('Config file saved...')
    print('FINISHED EVALUATION')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir',type=str, required=True,help='Evaluation Parameter Configuration File is not provided')
    args = parser.parse_args()
    main(args)
