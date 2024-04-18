import yaml 
import pandas
import matplotlib.pyplot as plt
import numpy as np

def eval_stat(df,mode='raw'):
    if mode =='raw':

        #Plot solve time
        plt.figure()
        plt.plot(np.arange(1,len(df['solve_time_exp_arr'])+1),df['solve_time_exp_arr'])
        plt.plot(np.arange(1,len(df['solve_time_policy_arr'])+1),df['solve_time_policy_arr'])
        # plt.plot(np.arange(1,len(df['NN_query_time'])+1),df['NN_query_time'])
        plt.legend(['Expert','Learned Policy','NN Query Time'])
        plt.ylabel('Solve Time (s)')
        plt.xlabel('Steps')
        plt.show()

        #plot feasibility
        plt.figure()
        plt.plot(np.arange(1,len(df['infeas_exp_arr'])+1),df['infeas_exp_arr'])
        plt.plot(np.arange(1,len(df['infeas_policy_arr'])+1),df['infeas_policy_arr'])
        plt.legend(['Expert','Learned Policy'])
        plt.ylabel('Infeasibility')
        plt.xlabel('Steps')
        plt.show()

        #plot vars kept, constr_kept
        plt.figure()
        # max?
        plt.plot(np.arange(1,len(df['vars_kept'])+1),df['vars_kept'])
        plt.plot(np.arange(1,len(df['const_kept'])+1),df['const_kept'])
        plt.legend(['# Variables','# Constraint'])
        plt.ylabel('Quantity')
        plt.xlabel('Steps')
        plt.show()

    else:
        exp_sol_time_avg = np.mean(df['solve_time_exp_arr'])
        exp_sol_time_std = np.std(df['solve_time_exp_arr'])
        pol_sol_time_avg = np.mean(df['solve_time_policy_arr'])
        pol_sol_time_std = np.std(df['solve_time_policy_arr'])

        print(f'Expert average solve time in a single trajectory: {exp_sol_time_avg}, std: {exp_sol_time_std}\nLearned Policy average solve time in a single trajectory: {pol_sol_time_avg}, std: {pol_sol_time_std}')


index = 3 #1-10
with open('data/configs/params_N14.yaml', 'r') as file:
    config = yaml.safe_load(file)

save_dir = config['root_dir'] + config['eval_save_dir']
model_name = (config['root_dir'] + config['model_dir']).split('/')[-1].split('.')[0]
solver = 'ipopt'
model_name = 'GRU_BC_Nov_30_21_09_06'
model_dir = save_dir +  'eval_' + str(index) + '_' + model_name + '_' + solver +'/'

df_eval = pandas.read_csv(model_dir + 'eval_' + str(index) + '_' + model_name + '_' + solver +'.csv',index_col=False)

eval_stat(df_eval,mode='raw')
eval_stat(df_eval,mode='avg')