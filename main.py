import numpy as np

from DDPG.agent import Agent
from DDPG.ddpg import DDPG
from DDPG.memory import Memory
from DDPG.noise import Noise
from hexapod_env import HexapodEnv

def _run(env):
    ddpg = DDPG(env.observation_space.shape, env.action_space.shape,
                env.action_space.high[0], env.action_space.low[0],
                0.001, 0.001, gamma=0.99, tau=0.001)
    memory = Memory(300000, env.observation_space.shape[0], env.action_space.shape[0])
    std_dev = 0.1
    noise = Noise(mean=np.zeros(env.action_space.shape),
                  std_deviation=float(std_dev) * np.ones(env.action_space.shape),
                  theta=0.15, dt=0.05)

    agent = Agent(env, ddpg, memory, noise, False)
    agent.run_episodes(max_episodes=5, max_steps=200)
    agent.save_results()
    agent.merge_recordings()

def first_experiment(disable_rendering):
    env = HexapodEnv(disable_rendering=disable_rendering, basic_rewards=True)
    _run(env)

def second_experiment(disable_rendering):
    env = HexapodEnv(disable_rendering=disable_rendering, basic_rewards=False)
    _run(env)

def main():
    run_experiment_num = 2
    if run_experiment_num == 1:
        first_experiment(disable_rendering=True)
    elif run_experiment_num == 2:
        second_experiment(disable_rendering=True)
    else:
        print("Wrong experiment number")

if __name__ == '__main__':
    main()