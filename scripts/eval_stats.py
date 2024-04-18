import os
import pandas as pd
import yaml
import argparse

def main(args):
    #Import simulation environment configuration parameters (must be consistent with the training config)
    with open(args.config_dir, 'r') as file:
        config = yaml.safe_load(file)
    
    dir_list = []
    if args.model_dir:
        model_name = args.model_dir.split('/')[-1].split('.')[0]
    root_dir = config['eval_save_dir']

    for dir in os.listdir(root_dir):
        if model_name in dir:
            dir_list.append(dir)
        
    counter = 0
    expert_collision = 0
    policy_collision = 0
    for dir in dir_list:
        folder_name = dir
        file_name = '_'.join(folder_name.split('_')[:-1]) + '.csv'
        file_path = root_dir + folder_name + '/' + file_name 
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path,index_col=False)
            if counter == 0:
                df_total = df
            else:
                df_total = pd.concat([df_total,df])
            counter += 1
        else:
            try:
                print(root_dir + dir + '/collision.txt')
                f = open(root_dir + dir + '/collision.txt',"r")
                collision_ind = f.read()
                if collision_ind == 'expert':
                    expert_collision += 1
                else:
                    policy_collision += 1
            except:
                pass
                # print(f'File doesn\'t exist. Skipping {file_path}')

    if 'df_total' in locals():
        avg_exp_solve_time = df_total['solve_time_exp_arr'].mean()
        std_exp_solve_time = df_total['solve_time_exp_arr'].std()

        avg_policy_solve_time = df_total['solve_time_policy_arr'].mean()
        std_policy_solve_time = df_total['solve_time_policy_arr'].std()

        avg_NN_query_time = df_total['NN_query_time'].mean()
        std_NN_query_time = df_total['NN_query_time'].std()
        avg_const_kept = df_total['const_kept'].mean()

        infeas_exp = df_total['infeas_exp_arr'].sum() / df_total['infeas_exp_arr'].size
        infeas_policy = df_total['infeas_policy_arr'].sum() / df_total['infeas_policy_arr'].size

        dt = 0.2
        norm_exp_task_time = (sum(df_total['solve_time_exp_arr'].isna()==0) * dt)
        norm_policy_task_time = sum(df_total['solve_time_policy_arr'].isna()==0) * dt / norm_exp_task_time

        print(f'{counter} Evaluation Runs Loaded')
        print(f'Expert collided in {expert_collision} runs')
        print(f'Policy collided in {policy_collision} runs out of {len(dir_list)} runs: {policy_collision/len(dir_list) * 100} %')
        print(f'Avg. Expert Solve Time {avg_exp_solve_time} (s), Std Expert Solve Time {std_exp_solve_time} (s)')
        print(f'Avg. Policy Solve Time {avg_policy_solve_time} (s), Std Policy Solve Time {std_policy_solve_time} (s)')
        print(f'Avg. NN Query Time {avg_NN_query_time} (s), Std. NN Query Time  {std_NN_query_time} (s)')
        print(f'Avg. const_kept: {avg_const_kept/624*100} %')
        print(f'Expert Infeas rate: {infeas_exp*100} %, Reduced SMPC Infeas rate: {infeas_policy*100} %')
        print(f'Avg Improvement (just solve time): {avg_exp_solve_time/avg_policy_solve_time} Times Faster')
        print(f'Avg Improvement (w/ NN query time): {avg_exp_solve_time/(avg_policy_solve_time + avg_NN_query_time)} Times Faster')

        print(f'Average Normalized Task Completion Time: (Full SMPC) {norm_exp_task_time}, (Reduced SMPC){norm_policy_task_time}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir',type=str, required=True,help='Evaluation Parameter Configuration File is not provided')
    parser.add_argument('-model_dir',type=str, required=False,help='Evaluation Model Path is not provided',default=None)
    args = parser.parse_args()
    main(args)