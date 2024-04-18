from bc import BC
import numpy as np
from infrastructure.utils import sample_trajectories
import pickle
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.utils import convert_listofrollouts
import torch as th
from datetime import datetime
import random
import os, logging

class dagger():
    def __init__(self,env,policy=None,collect_data=False,save_expert_data=False,logger=None, config=None, model_dir = None,policy_type ='GRU',normalize=True):
        self.env = env
        self.policy=policy
        self.model_dir = model_dir
        self.collect_data=collect_data
        self.save_expert_data=save_expert_data
        self.config=config
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.logger = logger
        self.policy_type = policy_type
        self.normalize = normalize
        if th.cuda.is_available():
            self.use_cuda = True 
        else:
            self.use_cuda = False        

        if isinstance(self.policy,list) and self.policy[0].pred_mode[1]=='tertiary':
            self.tertiary_l1 = True
        else:
            self.tertiary_l1 = False

    def collect_exp_data(self,seed=None):
        N_ROLLOUTS = self.config["N_ROLLOUTS"]
        max_path_length = self.config["max_path_length"]
        min_timesteps_per_batch = self.config["min_timesteps_per_batch"]
        total_envsteps = 0
        seed_ = None
        if self.env.reduced_mode:
            N_TV = 3
        else:
            N_TV = 5

        filepath = self.config['root_dir'] + self.config['exp_traj_save_dir'] + 'traffic_env_exp_demo' + '_N_' + str(self.env.N) + '_N_TV_' + str(N_TV) + '.pkl'

        '''
        Expert Trajectory Collection (Training Dataset: D)
        '''
        print(f'Saving dataset to {filepath}'.center(80,'*'))
        if self.collect_data:
            self.obs = None
            print('Expert Trajectory Collection Started'.center(80,'*'))
            if seed is not None:
                seeds = random.choices(range(1000),k=N_ROLLOUTS)
            for ep in range(int(N_ROLLOUTS)):
                print(f"Episode {ep}/{N_ROLLOUTS}".center(80,'-'))
                if seed is not None:
                    seed_ = seeds[ep]
                    print(f'Random Seed: {seed_}')
                    
                paths, envsteps_this_batch = sample_trajectories(self.env,policy=None,min_timesteps_per_batch=min_timesteps_per_batch,max_path_length=max_path_length,seed=seed_,use_cuda=self.use_cuda, dagger_mode=True)
                total_envsteps += envsteps_this_batch
                observations, actions, rewards, next_observations, terminals, dual_classes = (convert_listofrollouts(paths, concat_rew = True))

                if os.path.isfile(filepath):
                    with open(filepath, 'rb') as file:
                        paths = pickle.load(file)
                    data = {'observation':np.concatenate([paths['observation'],observations]),'reward': np.concatenate([paths['reward'],rewards]), 'action': np.concatenate([paths['action'],actions]), 'next_observation': np.concatenate([paths['next_observation'],next_observations]), "terminal": np.concatenate([paths['terminal'],terminals]), "dual_classes": np.concatenate([paths['dual_classes'],dual_classes])}   
                    with open(filepath, 'wb') as handle: #OVERWRITE THE FILE
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    data = {'observation':observations,'reward': rewards, 'action': actions, 'next_observation': next_observations, "terminal": terminals, "dual_classes": dual_classes}   
                    with open(filepath, 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        print('FINISHED COLLECTING DATA')      

    def learn(self, initial_train=False,seed=None):

        N_ROLLOUTS = self.config["N_ROLLOUTS"]
        N_EPOCHS_BC = self.config["N_EPOCHS_BC"]
        optimizer = self.config["optimizer"]
        lr = self.config["lr"]

        max_path_length = self.config["max_path_length"]
        min_timesteps_per_batch = self.config["min_timesteps_per_batch"]
        render = False
        total_envsteps = 0

        '''
        Expert Trajectory Collection (Training Dataset: D)
        '''
        print('Loading Expert Trajectory Data'.center(80,'*'))
        with open(self.config['expert_trajectory_data'],'rb') as file:
            paths = pickle.load(file)
        envsteps_this_batch = 0
        total_envsteps += envsteps_this_batch

        replay_buffer = ReplayBuffer(self.config['max_replay_buffer_size'])
        replay_buffer.obs = paths["observation"]; replay_buffer.acs = paths["action"]; replay_buffer.terminals = paths["terminal"]; replay_buffer.next_obs = paths["next_observation"]; replay_buffer.rews = paths['reward'] 
        replay_buffer.dual_classes = paths["dual_classes"]
        replay_buffer.set_weights()

        '''
        Train the policy on the training dataset {D} using the Behavior Cloning algorithm
        '''
        print('Executing BC Algorithm'.center(80,'*'))
        self.bc_learner = BC(policy=self.policy,optimizer=optimizer,optim_lr=lr, observation_space= self.env.observation_space, action_space=self.env.action_space, demonstrations=replay_buffer, rng = np.random.default_rng(0),device = self.device, batch_size=self.config['training_batch_size'],logger=self.logger,normalize=self.normalize, config=self.config, normalize_obs=self.config['normalize_obs'], dagger_mode=True, l1_dual_dim= [self.env.smpc.N-1, self.env.smpc.N_modes, self.env.smpc.N_TV], ca_dual_dim= [self.env.smpc.N-1, len(self.env.smpc.mode_map), self.env.smpc.N_TV])

        #Load model
        if self.model_dir is None:
            pass
        else:
            self.load(model_dir=self.model_dir)
            self.bc_learner.update_policy(self.policy)

        #Train the policy on the provided training dataset
        if initial_train:
            training_log = self.bc_learner.train(n_epochs=N_EPOCHS_BC, n_batches=self.config["training_batch_size"],binary_pred=True)
            print(f'Avg. Loss: {np.mean(self.bc_learner.training_loss)}\n Std Loss: {np.std(self.bc_learner.training_loss)}\n Min Loss: {min(self.bc_learner.training_loss)}\n Max Loss: {max(self.bc_learner.training_loss)}')
        
        dagger_filepath = self.config['expert_trajectory_data'].split('.')[0] + '_dagger.pkl'
        for itr in range(self.config['dagger_n_iter']):
            print(f"Iteration {itr}".center(80,'#'))
            '''
            Dataset Aggregation using querying the expert policy: {D U D_pi}
            '''
            print('Collecting data using the learned policy'.center(80,'*'))
            if self.save_expert_data:
                self.obs = None

            if seed is not None:
                seeds = random.choices(range(1000),k=N_ROLLOUTS)

            for ep in range(int(N_ROLLOUTS)):
                print(f"Episode {ep}/{N_ROLLOUTS}".center(80,'-'))
                if seed is not None:
                    seed_ = seeds[ep]
                    print(f'Random Seed: {seed_}')
                #dagger mode = True means don't discard the trajectory even when collisions occur
                paths, envsteps_this_batch = sample_trajectories(self.env,policy=self.bc_learner.policy,min_timesteps_per_batch=min_timesteps_per_batch,max_path_length=max_path_length,use_cuda=self.use_cuda,tertiary_l1=self.tertiary_l1,normalize_obs=self.config['normalize_obs'],seed=seeds[ep] if seed else None,dagger_mode=True)
                total_envsteps += envsteps_this_batch
                replay_buffer.add_rollouts(paths) #This will discard some(most) of the imported expert data

                #For saving expert trajectory data
                if self.save_expert_data:
                    observations, actions, rewards, next_observations, terminals, dual_classes = (convert_listofrollouts(paths, concat_rew = True))
                    if self.obs is None:
                        self.obs = observations
                        self.acs = actions
                        self.rews = rewards
                        self.next_obs = next_observations
                        self.terminals = terminals
                        self.dual_classes = dual_classes
                    else:
                        self.obs = np.concatenate([self.obs, observations])
                        self.acs = np.concatenate([self.acs, actions])
                        self.rews = np.concatenate(
                            [self.rews, rewards]
                        )
                        self.next_obs = np.concatenate([self.next_obs, next_observations])
                        self.terminals = np.concatenate([self.terminals, terminals])
                        self.dual_classes = np.concatenate([self.dual_classes, dual_classes])

                if self.save_expert_data:
                    if os.path.isfile(dagger_filepath):
                        with open(dagger_filepath, 'rb') as file:
                            paths = pickle.load(file)
                        data = {'observation':np.concatenate([paths['observation'],observations]),'reward': np.concatenate([paths['reward'],rewards]), 'action': np.concatenate([paths['action'],actions]), 'next_observation': np.concatenate([paths['next_observation'],next_observations]), "terminal": np.concatenate([paths['terminal'],terminals]), "dual_classes": np.concatenate([paths['dual_classes'],dual_classes])}   
                        with open(dagger_filepath, 'wb') as handle: #OVERWRITE THE FILE
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        data = {'observation':observations,'reward': rewards, 'action': actions, 'next_observation': next_observations, "terminal": terminals, "dual_classes": dual_classes}   
                        with open(dagger_filepath, 'wb') as handle:
                                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                            
            '''
            Train the policy on the aggregated Dataset: {D U D_pi}
            '''
            print('Executing BC Algorithm on the aggregated Dataset'.center(80,'*'))
            self.bc_learner.set_demonstrations(replay_buffer) #Update the demonstration data
            training_log = self.bc_learner.train(n_epochs=N_EPOCHS_BC, n_batches=self.config["training_batch_size"],binary_pred=True) #Run BC on the aggregated Dataset
            for key, value in training_log.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, int(itr*(self.config['N_EPOCHS_BC'])))
            print("Done logging...\n\n")
            self.logger.flush()
            print(f'Avg. Loss: {np.mean(self.bc_learner.training_loss)}\n Std Loss: {np.std(self.bc_learner.training_loss)}\n Min Loss: {min(self.bc_learner.training_loss)}\n Max Loss: {max(self.bc_learner.training_loss)}')

            #OPTIONAL (save):
            self.save(self.config['root_dir'] + self.config['model_save_dir'],itr)
            
        return self.bc_learner

    def save(self, model_save_dir,itr):
        if not self.file_path:
            now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
            self.file_path = model_save_dir + 'RAIDNET_DAGGER_' + now + '_'
            th.save({'model_state_dict': self.policy.state_dict(),
                        'optimizer_state_dict': self.bc_learner.optimizer.state_dict(),'config': self.config},
                    self.file_path + str(itr) + 'epoch'+ '.pt')
            
    def load(self, model_dir):
        save_file = None
        if os.path.exists(model_dir):
            save_file = model_dir
        if save_file is not None:
            checkpoint = th.load(save_file)
            print('Model loaded: {}'.format(save_file))
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.bc_learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return True
            logging.error('Can not find the model for {}'.format(model_dir))
        return False