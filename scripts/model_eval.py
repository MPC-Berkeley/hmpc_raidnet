import pickle
import yaml
import os
import torch as th
from infrastructure.policies import R_Tf_binary as R_Tf
from infrastructure.policies import mlp
from infrastructure.utils import observation_flatten, obs_normalize, to_tensor_var
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import warnings, argparse
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings( 'ignore' )
import pdb
import os

def main(args):
    with open(args.config_dir, 'r') as file:
        config = yaml.safe_load(file)

    compare_w_mlp = config['compare_with_mlp']

    #Create and register the custom gym environment
    gym.register(id='traffic_env-v0',entry_point='infrastructure.sim_gym:TrafficEnv')
    env = gym.make('traffic_env-v0',reduced_mode=config['reduced_mode'],N=config['N'],env_mode=0)
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
    device = th.device('cpu')

    #Load expert trajectory dataset
    with open(config['root_dir'] + config['eval_dataset_dir'],'rb') as file:
        data = pickle.load(file)

    #Initialize the NN policy
    policy=R_Tf(observation_dim, 2*observation_dim, ca_num, env.smpc.N-1, num_layers//2, hidden_size//2, lambda_dim=lambda_dim,lambda_ubd = lambda_ubd,pred_mode=['ca','tertiary','binary'],device = 'cpu')
    policy.to(device)
    if compare_w_mlp:
        ca_mlp_policy = mlp(observation_dim, hidden_size, ca_num, num_layers,lambda_dim,pred_mode=['ca','tertiary','binary'])
        ca_mlp_policy.to(device)

    #Load the model
    save_file = None
    if os.path.exists(args.model_dir):
        save_file = args.model_dir
    if save_file is not None:
        checkpoint = th.load(save_file)
        print('Model loaded: {}'.format(save_file))
        policy.load_state_dict(checkpoint['model_state_dict'])
    
    else:
        print('Can not find the model for {}'.format(config['model_dir']))


    #Load the mlp model for comparison
    if compare_w_mlp:
        if os.path.exists(config['root_dir'] + config['model_save_dir'] +config['ca_mlp_model_dir']):
                save_file = config['root_dir'] + config['model_save_dir']  + config['ca_mlp_model_dir']
        if save_file is not None:
            checkpoint = th.load(save_file)
            print('ca_mlp_model loaded: {}'.format(save_file))
            ca_mlp_policy.load_state_dict(checkpoint['model_state_dict'])
    '''
    Testing
    '''
    num_FN = 0
    num_FP = 0
    num_TP = 0
    num_TN = 0
    loss_fn = th.nn.BCEWithLogitsLoss(reduction='none',pos_weight=60*th.ones(policy.output_dim, device='cpu'))
    size =len(data['action'])
    obs = data['observation']
    ca_target  = th.tensor((data['action'][:,l1_num:]>1e-3).astype(float))

    wrong_pred    = th.zeros(ca_num)
    sample_target = th.ones(ca_num)
    max_loss = th.sum(loss_fn(wrong_pred,sample_target),dim=-1)
    print(f'Dataset Size: {size}')
    y_pred_arr = []
    nbins = 100
    
    for i in range(int(np.ceil(size / config['batch_size']))):

        pred_ca = policy(obs_normalize(to_tensor_var(obs[i*config['batch_size']:(i+1)*config['batch_size'],:], use_cuda=False)))
    
        if compare_w_mlp:
            pred_ca_mlp = ca_mlp_policy(obs_normalize(to_tensor_var(obs[i*config['batch_size']:(i+1)*config['batch_size'],:], use_cuda=False)))
        
        total_target = th.sum(ca_target)
        y_pred_arr.append(th.sigmoid(pred_ca.detach()).round().numpy().flatten())

        #test loss computation    
        loss = th.sum(loss_fn(pred_ca,ca_target[i*config['batch_size']:(i+1)*config['batch_size'],:]),dim=-1)
        if compare_w_mlp:
            loss_mlp = th.sum(loss_fn(pred_ca_mlp,ca_target[i*config['batch_size']:(i+1)*config['batch_size'],:]),dim=-1)

        if i == 0:
            loss_hist, loss_bin_edges = np.histogram((loss/max_loss).detach().numpy(),bins=np.linspace(0,1,nbins))
            if compare_w_mlp:
                loss_hist_mlp, loss_bin_edges_mlp = np.histogram((loss/max_loss).detach().numpy(),bins=np.linspace(0,1,nbins))
        else:
            hist, bin_edges = np.histogram((loss/max_loss).detach().numpy(),bins=np.linspace(0,1,nbins))
            assert np.all(loss_bin_edges == bin_edges)
            loss_hist += hist
            if compare_w_mlp:
                hist_mlp, bin_edges_mlp = np.histogram((loss_mlp/max_loss).detach().numpy(),bins=np.linspace(0,1,nbins))
                assert np.all(loss_bin_edges_mlp == bin_edges_mlp)
                loss_hist_mlp += hist_mlp

        #Confusion Matrix indices
        diff = th.sigmoid((pred_ca)).round()- ca_target[i*config['batch_size']:(i+1)*config['batch_size'],:]
        num_FN += th.sum(diff==-1)
        num_FP += th.sum(diff== 1)
        num_TP += th.sum((diff== 0) & (th.sigmoid((pred_ca)).round()==1.))
        num_TN += th.sum((diff == 0) & (th.sigmoid((pred_ca)).round()==0.))


    recall = num_TP / (num_TP+ num_FN)

    print(f'Avg. True Positives: {int(num_TP)}. Avg. positive predictive value: {th.sum(num_TP)/(num_TP + num_FP)}. Got {num_TP/total_target} correct')
    print(f'Avg. False Negatives: {int(num_FN)}.Avg. false negative rate: {num_FN /(num_FN+num_TP)}')
    print(f'Avg. False Positive: {int(num_FP)}, Avg. True Negative: {num_TN}, Avg. recall: {recall}')     

    #Plot test loss as a histogram
    fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 3]})
    fig.set_figheight(8)
    fig.set_figwidth(5)
    ax[0].bar(loss_bin_edges[:-1], loss_hist, width=np.diff(loss_bin_edges), edgecolor="black", align="edge",label=r'$\pi^{\text{RAIDN}}$')
    if compare_w_mlp:
        ax[0].bar(loss_bin_edges_mlp[:-1], loss_hist_mlp, width=np.diff(loss_bin_edges_mlp), edgecolor="black", alpha=0.5,align="edge",color='orange',label=r'$\pi^{\text{MLP}}$')
    ax[0].set_ylabel('#')
    ax[0].set_xlabel(r'Normalized $\ell(\pi,\tilde{\mu}^\star)$')

    #HARDCODED:
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(0,16000)
    ax[0].legend()

    if compare_w_mlp:
        axins = zoomed_inset_axes(ax[0], 6, loc=10,axes_kwargs={"aspect":0.0000015})
        axins.bar(loss_bin_edges[:-1], loss_hist, width=np.diff(loss_bin_edges), edgecolor="black", align="edge")
        axins.bar(loss_bin_edges_mlp[:-1], loss_hist_mlp, width=np.diff(loss_bin_edges_mlp), edgecolor="black", alpha=0.5, align="edge",color='orange')
        axins.set(xlim=[0, 0.1], ylim=[0, 15000])
        axins.set_yticks(np.arange(0, 15001, 5000))
        axins.set_xticks(np.arange(0, 0.11, 0.05))
        mark_inset(ax[0], axins, loc1=2, loc2=3, fc="none", ec="0.5")
    for axs in ax:
        axs.set_anchor('C')

    #Plot confusion matrix:
    y_true, y_pred = ca_target.flatten(), np.concatenate(y_pred_arr)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true',cmap='Blues',im_kw={'vmax': 1.})
    disp.plot(ax=ax[1],cmap='Blues',im_kw={'vmax': 1.},text_kw={'fontsize':12,'fontweight':'bold'})
    plt.title('Normalized Confusion Matrix on Test Dataset')
    plt.savefig(config['root_dir'] + "model_evaluation.png") 
    plt.show()
    print(f'Target Dataset... Active CA constraints: {sum(y_true)}. Total CA variables: {len(y_true)}. {sum(y_true)/len(y_true)*100} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir',type=str, required=True,help='Evaluation Parameter Configuration File is not provided')
    parser.add_argument('-model_dir',type=str, required=False,help='Evaluation Model Path is not provided',default=None)
    args = parser.parse_args()
    main(args)