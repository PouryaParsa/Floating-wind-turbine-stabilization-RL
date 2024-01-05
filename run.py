import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gym
from stable_baselines3 import PPO
import utils

def save_simulation_data(sim_df, directory, base_filename):
    i = 0
    while os.path.exists(os.path.join(directory, f"{base_filename}_{i}.csv")):
        i += 1
    sim_df.to_csv(os.path.join(directory, f"{base_filename}_{i}.csv"))

def plot_simulation_data(sim_df):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(sim_df['x_tf'], label='Fore-Aft')
    axs[0, 0].plot(sim_df['x_ts'], label='Side-Side')
    axs[0, 0].set_ylabel('Meters')
    axs[0, 0].set_title('Tower top displacements')
    axs[0, 0].legend()

    axs[0, 1].plot(sim_df['theta_p'] * (180 / np.pi), label='Pitch')
    axs[0, 1].plot(sim_df['theta_r'] * (180 / np.pi), label='Roll')
    axs[0, 1].set_ylabel('Degrees')
    axs[0, 1].set_title('Angles')
    axs[0, 1].legend()

    axs[1, 0].plot(sim_df['Fa_1'], label='Fa_1', linestyle='--')
    axs[1, 0].plot(sim_df['Fa_2'], label='Fa_2')
    axs[1, 0].plot(sim_df['Fa_3'], label='Fa_3', linestyle='--')
    axs[1, 0].plot(sim_df['Fa_4'], label='Fa_4')
    axs[1, 0].set_ylabel('[N]')
    axs[1, 0].set_title('Inputs')
    axs[1, 0].legend()

    axs[1, 1].plot(sim_df['x_1'], label='x_1', linestyle='--')
    axs[1, 1].plot(sim_df['x_2'], label='x_2')
    axs[1, 1].plot(sim_df['x_3'], label='x_3', linestyle='--')
    axs[1, 1].plot(sim_df['x_4'], label='x_4')
    axs[1, 1].set_ylabel('Meters')
    axs[1, 1].set_title('DVA displacements')
    axs[1, 1].legend()

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument('--agent', help='Path to agent .zip file.')
    parser_group.add_argument('--lqr', help='Use LQR controller', action='store_true')
    parser.add_argument('--time', type=int, default=50, help='Max simulation time (seconds).')
    parser.add_argument('--plot', help='Show plot of states after simulation', action='store_true')
    args = parser.parse_args()

    env = gym.make("TurbineStab-v0")

    if args.lqr:
        sim_df = utils.simulate_episode(env=env, agent=None, max_time=args.time, lqr=True)
        save_simulation_data(sim_df, "logs/LQR_simulations", "_simdata_lqr")
    else:
        agent = PPO.load(args.agent)
        sim_df = utils.simulate_episode(env=env, agent=agent, max_time=args.time, lqr=False)
        agent_path_list = args.agent.split(os.path.sep)
        simdata_dir = os.path.join("logs", agent_path_list[-3], "sim_data")
        os.makedirs(simdata_dir, exist_ok=True)
        save_simulation_data(sim_df, simdata_dir, f"_simdata_{agent_path_list[-1][0:-4]}")

    env.close()

    if args.plot:
        plot_simulation_data(sim_df)

if __name__ == "__main__":
    main()
