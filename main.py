import numpy as np
import sys

from tensorflow.python.ops.gen_experimental_dataset_ops import experimental_csv_dataset

from DDPG.agent import Agent
from DDPG.ddpg import DDPG
from DDPG.memory import Memory
from DDPG.noise import Noise
from hexapod_env import HexapodEnv

def _run(env, directory_name):
    ddpg = DDPG(env.observation_space.shape, env.action_space.shape,
                env.action_space.high[0], env.action_space.low[0],
                0.001, 0.001, gamma=0.99, tau=0.001)
    memory = Memory(500_000, env.observation_space.shape[0], env.action_space.shape[0], batch_size=4096)
    #std_dev = 0.1
    std_dev = 0.2
    # noise = Noise(mean=np.zeros(env.action_space.shape),
    #               std_deviation=float(std_dev) * np.ones(env.action_space.shape),
    #               theta=0.15, dt=0.05)
    noise = Noise(mean=np.zeros(env.action_space.shape),
                  std_deviation=float(std_dev) * np.ones(env.action_space.shape),
                  theta=0.15, dt=0.05, decay_constant=0.85)

    agent = Agent(env, ddpg, memory, noise, False)
    agent.run_episodes(max_episodes=20_000, max_steps=300, warm_up_episodes=200, preload_warmup=True)
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
    env = HexapodEnv(disable_rendering=disable_rendering, rewards=2)
    _run(env, directory_name)

def experiment_test(directory_name, disable_rendering):
    env = HexapodEnv(disable_rendering=disable_rendering, rewards=2)
    _run_test(env)

def main():
    if len(sys.argv) == 3:
        """ Training """
        if sys.argv[2] == "False":
            directory_name = sys.argv[1] if len(sys.argv) > 2 else None
            experiment(directory_name, disable_rendering=True)
        else:
            """ Testing """
            experiment_test(directory_name=None, disable_rendering=False)

if __name__ == '__main__':
    main()