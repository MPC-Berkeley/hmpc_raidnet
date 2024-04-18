import gymnasium as gym
import yaml, argparse
from dagger import dagger
import warnings
warnings.filterwarnings( 'ignore' )

def main(args):
    #Import data collection configuration parameters
    with open(args.config_dir, 'r') as file:
        config = yaml.safe_load(file)

    #Create and register the custom gym environment
    gym.register(id='traffic_env-v0',entry_point='infrastructure.sim_gym:TrafficEnv')
    env = gym.make('traffic_env-v0',reduced_mode=config['reduced_mode'],N=config['N'],use_ttc=True, env_mode = 0)
    reduced_mode = config['reduced_mode']; N = config['N']
    print(f'Initializing traffic_env-v0 with reduced_mode: {reduced_mode}, N: {N}')
    obs, _ = env.reset()

    #Initialize dagger_learner
    collect_data = True
    dagger_learner = dagger(env,policy=None,collect_data=collect_data,save_expert_data=True,config=config)

    #Collect expert trajectory data
    dagger_learner.collect_exp_data(seed=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir',type=str, required=True,help='Evaluation Parameter Configuration File is not provided')
    args = parser.parse_args()
    main(args)