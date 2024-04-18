import yaml 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def eval_stat(df,mode='raw',save_dir=None):
    traj_ind = df['traj'][0]
    df = df.loc[df['traj']==traj_ind]

    if mode =='raw':
        #Plot solve time
        plt.figure()
        plt.plot(np.arange(1,len(df['solve_time_exp_arr'])+1),df['solve_time_exp_arr'])
        plt.plot(np.arange(1,len(df['solve_time_policy_arr'])+1),df['solve_time_policy_arr'])
        plt.plot(np.arange(1,len(df['NN_query_time'])+1),df['NN_query_time'])
        plt.legend(['Full SMPC','Reduced SMPC','NN Query'],loc='upper right')
        plt.ylabel('Solve Time (s)')
        plt.xlabel('Steps')
        plt.savefig(save_dir + "/solve_time_comparison.png") 
        plt.show()

        #plot feasibility
        plt.figure()
        plt.plot(np.arange(1,len(df['infeas_exp_arr'])+1),df['infeas_exp_arr'],'x')
        plt.plot(np.arange(1,len(df['infeas_policy_arr'])+1),df['infeas_policy_arr'],'x')
        plt.legend(['Expert','Learned Policy'])
        plt.ylabel('Infeasibility')
        plt.xlabel('Steps')
        plt.savefig(save_dir+"/traj_feasibility.png") 
        plt.show()

        #plot vars kept, constr_kept
        vars_kept_data = (df['vars_kept']/208 * 100).to_numpy()
        const_kept_data = (df['const_kept']/624 * 100).to_numpy()

        plt.figure()
        if args.do_var_screening:
            plt.plot(np.arange(1,len(df['vars_kept'])+1),vars_kept_data)
        plt.plot(np.arange(1,len(df['const_kept'])+1),const_kept_data)
        if args.do_var_screening:
            plt.legend(['Variables','Constraint'])
        plt.ylabel('% Kept')
        plt.xlabel('Steps')
        plt.savefig(save_dir+"/var_constr_screening.png") 
        plt.show()

    else:
        exp_sol_time_avg = np.mean(df['solve_time_exp_arr'])
        exp_sol_time_std = np.std(df['solve_time_exp_arr'])
        pol_sol_time_avg = np.mean(df['solve_time_policy_arr'])
        pol_sol_time_std = np.std(df['solve_time_policy_arr'])

        NN_query_time_avg = np.mean(df['NN_query_time'])
        NN_query_time_std = np.std(df['NN_query_time'])
        infeas_policy = np.sum(df['infeas_policy_arr']) / len(df['infeas_policy_arr'])
        infeas_exp    = np.sum(df['infeas_exp_arr']) / len(df['infeas_exp_arr'])

        save_data = {'Avg Expert Solve Time': exp_sol_time_avg, 'Std Expert Solve Time': exp_sol_time_std, 'Avg Reduced SMPC Solve Time': pol_sol_time_avg, 'Std Reduced SMPC Solve Time': pol_sol_time_std, 'NN_query_time_avg': NN_query_time_avg, 'NN_query_time_std': NN_query_time_std }
        save_df = pd.DataFrame(save_data, index=[0])
        save_df.to_csv(save_dir + '/solve_time_stats.csv')
        print(f'Expert average solve time in a single trajectory: {exp_sol_time_avg.round(3)}, std: {exp_sol_time_std.round(3)}\nReduced SMPC average solve time in a single trajectory: {pol_sol_time_avg.round(3)}, std: {pol_sol_time_std.round(3)}')
        print(f'Average NN Query Time in a single trajectory: {NN_query_time_avg.round(3)}, std: {NN_query_time_std.round(3)}')
        print(f'Average Feasibility % Expert: {(infeas_exp * 100).round(3)} %, Learned Policy: {(infeas_policy * 100).round(3)} %')

def main(args):
    df_eval = pd.read_csv(args.eval_dir,index_col=False)
    eval_stat(df_eval,mode='raw',save_dir='/'.join(args.eval_dir.split('/')[:-1]))
    print(f'Solver: {args.solver}' )
    eval_stat(df_eval,mode='avg',save_dir='/'.join(args.eval_dir.split('/')[:-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-solver',type=str, required=True, default='grb')
    parser.add_argument('-eval_dir',type=str, required=True)
    parser.add_argument('-do_var_screening',type=bool, default=False)
    args = parser.parse_args()
    main(args)