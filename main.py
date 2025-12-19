import os.path
import shutil

import numpy as np
import sys

from tensorflow.python.ops.gen_experimental_dataset_ops import experimental_csv_dataset

from DDPG.agent import Agent
# from DDPG.ddpg import DDPG
from DDPG.td3 import TD3
from DDPG.memory import Memory
from DDPG.noise import Noise
from hexapod_env import HexapodEnv

def _run(env, directory_name):
    td3 = TD3(env.observation_space.shape, env.action_space.shape,
                env.action_space.high[0], env.action_space.low[0],
                0.0001, 0.001, gamma=0.99, tau=0.005)
    memory = Memory(1_000_000, env.observation_space.shape[0], env.action_space.shape[0], batch_size=256)


    noise = Noise(std_deviation=float(0.3),
                  theta=0.15, dt=0.05, decay_constant=0.98, noise_shape=env.action_space.shape)

    agent = Agent(env, td3, memory, noise, False, directory_name)
    agent.run_episodes(max_episodes=20_000, max_steps=20, warm_up_memory=3_000, preload_warmup=False)
    agent.save_results(directory_name)
    agent.merge_recordings()


def _run_test(env):
    ddpg = DDPG(env.observation_space.shape, env.action_space.shape,
                env.action_space.high[0], env.action_space.low[0],
                0.001, 0.001, gamma=0.99, tau=0.001)

    agent = Agent(env, ddpg, None, None, False)
    agent.load_weights()

    agent.run_test(max_episodes=2, max_steps=1000)

def experiment(directory_name, disable_rendering):
    while True:
        if os.path.exists(directory_name):
            print(f"Directory {directory_name} already exists. Do you want to overwrite it? (y/n)")
            if input().lower() == "y":
                os.makedirs(directory_name, exist_ok=True)
                break
            else:
                directory_name = input(f"Plase provide new directory name: ")
        else:
            break

    env = HexapodEnv(disable_rendering=disable_rendering, rewards=2)
    _run(env, directory_name)

def experiment_test(directory_name, disable_rendering):
    env = HexapodEnv(disable_rendering=disable_rendering, rewards=2)
    _run_test(env)

def main():
    if len(sys.argv) == 3:
        """ Training """
        if sys.argv[2] == "Train":
            directory_name = sys.argv[1] if len(sys.argv) > 2 else None
            print(directory_name)
            experiment(directory_name, disable_rendering=True)
        elif sys.argv[2] == "Test":
            """ Testing """
            experiment_test(directory_name=None, disable_rendering=False)

if __name__ == '__main__':
    main()