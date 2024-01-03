import os
import multiprocessing
import argparse
import json
from time import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from gym_turbine import reporting
from gym_turbine.reporting import ReportingCallback, TensorboardCallback

class TrainingConfig:
    def __init__(self, timesteps=500000, agent=None, note=None, no_reporting=False):
        self.timesteps = timesteps
        self.agent = agent
        self.note = note
        self.no_reporting = no_reporting

def setup_experiment():
    experiment_id = str(int(time())) + 'ppo'
    agents_dir = os.path.join('logs', experiment_id, 'agents')
    os.makedirs(agents_dir, exist_ok=True)
    report_dir = os.path.join('logs', experiment_id, 'training_report')
    tensorboard_log = os.path.join('logs', experiment_id, 'tensorboard')
    return experiment_id, agents_dir, report_dir, tensorboard_log

def create_environment():
    num_cpus = multiprocessing.cpu_count()
    env = make_vec_env('TurbineStab-v0', n_envs=num_cpus, vec_env_cls=SubprocVecEnv)
    config_path = os.path.join('logs', experiment_id, "Note.txt")
    with open(config_path, "a") as config_file:
        config_file.write("env_config: " + json.dumps(env.get_attr('config')[0]))
        if args.note:
            config_file.write(args.note)
    return env

def create_callbacks(report_dir, tensorboard_log, no_reporting):
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=agents_dir)
    reporting_callback = ReportingCallback(report_dir=report_dir, verbose=True)
    tensorboard_callback = TensorboardCallback(verbose=True)
    if no_reporting:
        return CallbackList([checkpoint_callback, tensorboard_callback])
    else:
        return CallbackList([checkpoint_callback, reporting_callback, tensorboard_callback])

def create_agent(env, agent_path=None):
    if agent_path:
        agent = PPO.load(agent_path, env=env, verbose=True, tensorboard_log=tensorboard_log)
    else:
        agent = PPO('MlpPolicy', env, verbose=True, tensorboard_log=tensorboard_log)
    return agent

def main(args):
    experiment_id, agents_dir, report_dir, tensorboard_log = setup_experiment()
    env = create_environment()
    callback = create_callbacks(report_dir, tensorboard_log, args.no_reporting)
    agent = create_agent(env, args.agent)

    agent.learn(total_timesteps=args.timesteps, callback=callback)

    agent_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps))
    agent.save(agent_path)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000, help='Number of timesteps to train the agent. Default=500000')
    parser.add_argument('--agent', help='Path to the RL agent to continue training from.')
    parser.add_argument('--note', type=str, default=None, help="Note with additional info about training")
    parser.add_argument('--no_reporting', help='Skip reporting to increase framerate', action='store_true')
    args = parser.parse_args()

    config = TrainingConfig(timesteps=args.timesteps, agent=args.agent, note=args.note, no_reporting=args.no_reporting)
    experiment_id, agents_dir, report_dir, tensorboard_log = setup_experiment()
    env = create_environment()
    callback = create_callbacks(report_dir, tensorboard_log, config.no_reporting)
    agent = create_agent(env, config.agent)

    agent.learn(total_timesteps=config.timesteps, callback=callback)

    agent_path = os.path.join(agents_dir, "last_model_" + str(config.timesteps))
    agent.save(agent_path)

    env.close()

