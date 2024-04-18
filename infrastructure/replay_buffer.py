from infrastructure.utils import *
from collections import Counter
import torch as th
from infrastructure.utils import convert_listofrollouts

class ReplayBuffer(th.utils.data.Dataset):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None
        self.dual_classes = None
        self.num_classes=4
        self.class_weights = np.zeros(self.num_classes)

        
        

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0
        
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx=idx.tolist()
        
        return self.obs[idx], self.acs[idx], self.dual_classes[idx]
    

    def set_weights(self):
        
        if self.obs is None:
            raise(ValueError('No data for weight calculation!'))
            
        self.class_weights = np.zeros(self.num_classes)

        class_count = Counter(self.dual_classes)

        for i in range(self.num_classes):
            if i in class_count:
                self.class_weights[i] = class_count[i]*(1-0.9*int(i==1 or i==2 or i ==3))
        total =  np.sum(self.class_weights)

        self.class_weights = (total - self.class_weights)/total
        self.dataset_weights = np.zeros(self.max_size)

        # for i in range(self.max_size):
            # self.dataset_weights[i] = self.class_weights[self.dual_classes[i]]
        self.dataset_weights = self.class_weights[self.dual_classes]
        
    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        self.class_weights = np.zeros(self.num_classes)
        
        for path in paths:
            self.paths.append(path)
           

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals, dual_classes = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.dual_classes = dual_classes[-self.max_size:]


        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            self.dual_classes = np.concatenate(
                [self.dual_classes, dual_classes]
            )[-self.max_size:]
        
        #Update the class weights

        self.set_weights()



        
