import argparse
import pickle
import numpy as np

def main(args):
    '''
    Script to combine two datasets into one dataset
    '''
    with open(args.data1_dir,'rb') as file:
        data1 = pickle.load(file)

    with open(args.data2_dir,'rb') as file:
        data2 = pickle.load(file)

    combined_data = {'observation':np.concatenate([data1['observation'],data2['observation']]),'reward': np.concatenate([data1['reward'],data2['reward']]), 'action': np.concatenate([data1['action'],data2['action']]), 'next_observation': np.concatenate([data1['next_observation'],data2['next_observation']]), "terminal": np.concatenate([data1['terminal'],data2['terminal']]), "dual_classes": np.concatenate([data1['dual_classes'],data2['dual_classes']])}   

    with open(args.data1_dir, 'wb') as handle: #OVERWRITE THE FILE (data1)
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data1_dir',type=str, required=True)
    parser.add_argument('-data2_dir',type=str, required=True)
    args = parser.parse_args()
    main(args)