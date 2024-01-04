import numpy as np
from scipy.io import loadmat
from pandas import DataFrame

def simulate_episode(env, agent, max_time, lqr=False):
    state_labels = [f"x_{i}" for i in range(1, 12)]
    state_dot_labels = [f"x_{i}_dot" for i in range(1, 12)]
    input_labels = [f"Fa_{i}" for i in range(1, 5)]
    reward_label = "reward"
    labels = np.hstack(["time", state_labels, state_dot_labels, input_labels, reward_label])

    if lqr:
        K = loadmat('gym_turbine\\utils\\Ksys.mat')["Ksys_lqr"]
        # K = loadmat('gym_turbine\\utils\\LQR_params.mat')["K"]

    done = False
    env.reset()
    
    while not done and env.t_step < max_time/env.step_size:
        if lqr:
            action = (-K.dot(env.turbine.state))/ss.max_input
        else:
            action, _states = agent.predict(env.observation, deterministic=True)
        _, _, done, _ = env.step(action)

    time = np.array(env.episode_history['time']).reshape((env.t_step, 1))
    last_reward = np.array(env.episode_history['last_reward']).reshape((env.t_step, 1))
    sim_data = np.hstack([time, env.episode_history['states'], env.episode_history['states_dot'],
                          env.episode_history['input'], last_reward])
    df = DataFrame(sim_data, columns=labels)
    return df

