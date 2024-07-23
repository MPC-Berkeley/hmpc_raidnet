<div align="center">

# RAID-Net: Recurrent Attention-based for Interaction Duals Network
This repository contains the implementation of the paper <em>"Scalable Multi-modal Model Predictive Control via Duality-based Interaction Predictions"</em> accepted at 2024 IEEE Intelligent Vehicles Symposium 

[Hansung Kim (hansung@berkeley.edu)](https://github.com/hansungkim98122) &emsp; [Siddharth Nair (siddharth_nair@berkeley.edu)](https://shn66.github.io/) &emsp; [Francesco Borrelli](https://me.berkeley.edu/people/francesco-borrelli/)   

![](https://img.shields.io/badge/language-python-blue)
<a href='https://arxiv.org/abs/2402.01116'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

# Demonstration Video:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-pRiOnPb9_c/0.jpg)](https://www.youtube.com/watch?v=-pRiOnPb9_c)
</div>


# Example Simulation Results
<div align="center">
<img src="https://github.com/shn66/L4SMPC/blob/IV2024/assets/eval_expert.gif" width="400" /> <img src="https://github.com/shn66/L4SMPC/blob/IV2024/assets/eval_NN.gif" width="400">
  
Baseline (Full MPC with collision avoidance constraints imposed for all multi-modal predictions)
Baseline (Full MPC with collision avoidance constraints imposed for all multi-modal predictions)

<img src="https://github.com/shn66/L4SMPC/blob/IV2024/assets/results_table.png" width="500">

<strong>>12x Improvement in computation time of the motion planner!</strong>

</div>



# Installation
## Clone repository
```
git clone https://github.com/MPC-Berkeley/hmpc_raidnet.git
```
## Build environment
```
cd hmpc_raidnet
conda env create -n raidnet -f environment.yml
conda activate raidnet
```
## Install dependency
```
pip install -e .
```
## Installing solvers
Running the simulator requires Gurobi and IPOPT. For installing Gurobi, find the link <a href='https://www.gurobi.com/academia/academic-program-and-licenses/'>here</a>. For installing IPOPT, 
```
wget -N -q "https://github.com/thomasfork/ipopt_linux/raw/main/ipopt-linux64.zip"
unzip -o -q ipopt-linux64
```
We provide a shell script 'grbpath.sh' which you can run in your terminal to export appropriate GUROBI related envrionment variables. Modify 'grbpath.sh' file to the paths in your machine.
```
source grbpath.sh
```

# Training RAID-Net
## Configuration:
The default configuration file is data/configs/params_N14.yaml. Change the root directory in the configuration files to appropriate paths in your local machine manually.
## Data collection:
```
python scripts/data_collect.py -config_dir <path_to_parameter_config_yaml>
python scripts/data_collect.py -config_dir ./data/configs/params_N14.yaml 
```
Alternatively, you can download a training <a href='https://drive.google.com/drive/folders/1BG3VQjl3Fv6RSOZcRV-xnOFao8mKMrGw?usp=drive_link'>dataset</a> using the default simulation environment parameters and data collection configurations (data/configs/params_N14.yaml). Modify the configuration file to change the parameters as needed. 

If you choose to download and use the provided dataset, move the dataset .pkl files as shown below
```
├── assets
│   ├── ...
├── data
│   ├── configs
│   │   ├── ...
│   ├── expert_data
│   │   ├── <move data here>
│   ├── logs
│   │   ├── ...
│   ├── models
│   │   ├── ...
├── infrastructure
│   ├── ...
├── scripts
│   ├── ...
├── ...
```

## Running behavior cloning:
Before training, modify the **expert_trajectory_data path** in ./data/configs/*yaml file that you are using. For training a RAID-Net Policy, run
```
python scripts/run_bc.py -config_dir <path_to_parameter_config_yaml>
python scripts/run_bc.py -config_dir ./data/configs/params_N14.yaml 
```
For training a Simple MLP Policy for comparison,
```
python scripts/run_bc_mlp.py -config_dir <path_to_parameter_config_yaml>
python scripts/run_bc_mlp.py -config_dir ./data/configs/params_N14.yaml 
```
(WIP) For running DAgger,
```
python scripts/run_dagger.py -config_dir <path_to_parameter_config_yaml>
python scripts/run_dagger.py -config_dir ./data/configs/params_N14.yaml 
```
Alternatively, you can download the pre-trained RAID-Net for N=14 and Num TV = 3 in our custom traffic intersection simulation environment <a href='https://drive.google.com/drive/folders/1BRlaWDZPhwlfURWnMpRHiwTLyzjBN-4q?usp=sharing'>here</a>. After downloading the model, move the model files .pt as shown below

```
├── assets
│   ├── ...
├── data
│   ├── configs
│   │   ├── ...
│   ├── expert_data
│   │   ├── ...
│   ├── logs
│   │   ├── ...
│   ├── models
│   │   ├── <move models here>
├── infrastructure
│   ├── ...
├── scripts
│   ├── ...
├── ...
```

# Evaluation
Evaluation parameters can be modified in the ./data/configs/*.yaml files. Specifically, you must change the **root_dir** variable in the config files to an appropriate path in your machine. Also, you can set the planner horizon length N, model names, and other training and environmental parameters as needed. 
```
python scripts/closedloop_eval.py -config_dir <path_to_parameter_config_yaml>
python scripts/closedloop_eval.py -config_dir ./data/configs/params_N14.yaml 
```
The evaluation results will be recorded in ./data/models/evaluation/ including mp4 files of the closed-loop simulation results. After running closedloop_eval, the statistics of the evaluation results can be obtained by running
```
python scripts/eval_stats.py -config_dir <path_to_parameter_config_yaml> -model_dir <path_to_model: optional>
python scripts/eval_stats.py -config_dir ./data/configs/params_N14.yaml -model_dir ./data/models/RAIDNET_BC_Jan_21_09_13_20_CA.pt
```
Furthermore, the RAID-Net model (as a classifier) itself can be evaluated on a test dataset to obtain the confusion matrix and normalized classification loss of the model by running
```
python scripts/model_eval.py -config_dir <path_to_parameter_config_yaml> -model_dir <path_to_model: optional>
python scripts/model_eval.py -config_dir ./data/configs/model_eval.yaml -model_dir ./data/models/RAIDNET_BC_Jan_21_09_13_20_CA.pt
```

<!-- # Visualizing Evaluation Results
```
python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_3_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb_887_no_l1/eval_3_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb.csv

python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_4_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb_410_no_l1/eval_4_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb.csv

python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_4_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb_410/eval_4_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb.csv
```

python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_7_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb_245_no_l1/eval_7_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb.csv

python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_7_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb_245/eval_7_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_grb.csv


python plot_figures.py -solver ipopt -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_0_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_ipopt_470/eval_0_GRU_BC_Jan_16_22_39_18_L1_Noneepoch_ipopt.csv


python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_1_GRU_BC_Jan_17_06_42_33_L1_300epoch_grb_728/eval_1_GRU_BC_Jan_17_06_42_33_L1_300epoch_grb.csv

python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_5_GRU_BC_Jan_17_06_42_33_L1_300epoch_grb_716/eval_5_GRU_BC_Jan_17_06_42_33_L1_300epoch_grb.csv

python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_5_GRU_BC_Jan_17_06_42_33_L1_300epoch_grb_716_with_l1/eval_5_GRU_BC_Jan_17_06_42_33_L1_300epoch_grb.csv

python plot_figures.py -solver grb -eval_dir /home/mpc/L4SMPC/data/models/evaluation/eval_0_GRU_BC_Jan_17_21_57_33_L1_100epoch_ipopt_470/eval_0_GRU_BC_Jan_17_21_57_33_L1_100epoch_ipopt.csv -->

# Combining Datasets (if needed)
If you need to combine two datasets, you can use combine_dataset.py
```
python scripts/combine_dataset.py -data1_dir <path_to_dataset1> -data2_dir <path_to_dataset2>
```

