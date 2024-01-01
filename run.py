import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_turbine
from stable_baselines3 import PPO
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help='Path to agent .zip file.')
    parser.add_argument('--lqr', help='Use LQR controller', action='store_true')
    parser.add_argument('--time', type=int, default=50, help='Max simulation time (seconds).')
    parser.add_argument('--plot', help='Show plot of states after simulation', action='store_true')
    args = parser.parse_args()

    env = gym.make("TurbineStab-v0")

    if args.lqr:
        sim_df = utils.simulate_episode(env=env, agent=None, max_time=args.time, lqr=True)
        save_sim_data(sim_df, "LQR_simulations", "_simdata_lqr")
    else:
        agent = PPO.load(args.agent)
        sim_df = utils.simulate_episode(env=env, agent=agent, max_time=args.time, lqr=False)
        agent_path_list = args.agent.split(os.path.sep)
        save_sim_data(sim_df, os.path.join("logs", agent_path_list[-3], "sim_data"), f"_simdata_{agent_path_list[-1][0:-4]}")

    env.close()

    if args.plot:
        plot_states(sim_df)

def save_sim_data(sim_df, directory, file_prefix):
    os.makedirs(directory, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(directory, f"{file_prefix}_{i}.csv")):
        i += 1
    sim_df.to_csv(os.path.join(directory, f"{file_prefix}_{i}.csv"))

def plot_states(sim_df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(sim_df['x_tf'], label='Fore-Aft')
    ax1.plot(sim_df['x_ts'], label='Side-Side')
    ax1.set_ylabel('Meters')
    ax1.set_title('Tower top displacements')
    ax1.legend()

    ax2.plot(sim_df['theta_p']*(180/np.pi), label='Pitch')
    ax2.plot(sim_df['theta_r']*(180/np.pi), label='Roll')
    ax2.set_ylabel('Degrees')
    ax2.set_title('Angles')
    ax2.legend()

    ax3.plot(sim_df['Fa_1'], label='Fa_1', linestyle='--')
    ax3.plot(sim_df['Fa_2'], label='Fa_2')
    ax3.plot(sim_df['Fa_3'], label='Fa_3', linestyle='--')
    ax3.plot(sim_df['Fa_4'], label='Fa_4')
    ax3.set_ylabel('[N]')
    ax3.set_title('Inputs')
    ax3.legend()

    ax4.plot(sim_df['x_1'], label='x_1', linestyle='--')
    ax4.plot(sim_df['x_2'], label='x_2')
    ax4.plot(sim_df['x_3'], label='x_3', linestyle='--')
    ax4.plot(sim_df['x_4'], label='x_4')
    ax4.set_ylabel('Meters')
    ax4.set_title('DVA displacements')
    ax4.legend()

