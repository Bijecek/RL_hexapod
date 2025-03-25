import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers.legacy import Adam
#tf.keras.__version__ = "2.11.0"
#from rl.agents.ddpg import DDPGAgent
#from rl.memory import SequentialMemory
#from rl.random import OrnsteinUhlenbeckProcess

#import os
#os.environ['COPPELIASIM_ROOT'] = '/mnt/c/Users/sisin/Desktop/Seminarka/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
#os.environ['LD_LIBRARY_PATH'] = '/mnt/c/Users/sisin/Desktop/Seminarka/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/mnt/c/Users/sisin/Desktop/Seminarka/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'

# print(os.environ.get('COPPELIASIM_ROOT'))
# print(os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH'))
# print(os.environ.get('LD_LIBRARY_PATH'))


from DDPG.Agent import Agent
from DDPG.DDPG import DDPG
from DDPG.Memory import Memory
from DDPG.Noise import Noise
from HexapodEnv import HexapodEnv
import gymnasium as gym

def main():
    num_processes = 4  # Number of parallel simulations
    processes = []

    for i in range(num_processes):
        p = multiprocessing.Process(target=run_simulation, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    # env = HexapodEnv()
    #
    # # TODO: UNCOMMENT
    # # env = gym.make("Pendulum-v1", render_mode="human")
    # # num_states = env.observation_space.shape[0]
    # # num_actions = env.action_space.shape[0]
    # # upper_bound = env.action_space.high[0]
    # # lower_bound = env.action_space.low[0]
    # # print("num_states:", num_states)
    # # print("num_actions:", num_actions)
    # # print("Max Value of Action ->  {}".format(upper_bound))
    # # print("Min Value of Action ->  {}".format(lower_bound))
    #
    # ddpg = DDPG(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], env.action_space.low[0], 0.001 ,0.001, gamma=.99, tau=0.001)
    # memory = Memory(500000, env.observation_space.shape[0], env.action_space.shape[0])
    # #std_dev = 0.1
    # std_dev = 0.1
    # #noise = Noise(mean=np.zeros(env.action_space.shape), std_deviation=float(std_dev) * np.ones(env.action_space.shape), theta=0.15, dt=0.001)
    # noise = Noise(mean=np.zeros(env.action_space.shape), std_deviation=float(std_dev) * np.ones(env.action_space.shape),
    #               theta=0.15, dt=0.05)
    #
    #
    # agent = Agent(env, ddpg, memory, noise)
    # agent.run_episodes(2000)
    # agent.save_results()


def run_simulation(process_id):
    print(f"Process {process_id} starting...")
    env = HexapodEnv()

    # Initialize DDPG and components with unique names if needed
    ddpg = DDPG(env.observation_space.shape, env.action_space.shape,
                env.action_space.high[0], env.action_space.low[0],
                0.001, 0.001, gamma=.99, tau=0.001)
    memory = Memory(500000, env.observation_space.shape[0], env.action_space.shape[0])
    std_dev = 0.1
    noise = Noise(mean=np.zeros(env.action_space.shape),
                  std_deviation=float(std_dev) * np.ones(env.action_space.shape),
                  theta=0.15, dt=0.05)

    agent = Agent(env, ddpg, memory, noise)
    agent.run_episodes(1000)


if __name__ == '__main__':
    #print("Hi")
    main()

# def main():
#     env = HexapodEnv()
#
#     # Number of actions (18 joints)
#     nb_actions = env.action_space.shape[0]
#
#     # ======================== ACTOR NETWORK =========================
#     actor_input = Input(shape=(1,) + env.observation_space.shape)  # State input
#     x = Flatten()(actor_input)
#     x = Dense(256, activation="relu")(x)
#     x = Dense(128, activation="relu")(x)
#     x = Dense(64, activation="relu")(x)
#     actor_output = Dense(nb_actions, activation="tanh")(x)  # Outputs actions in range (-1,1)
#     actor = Model(inputs=actor_input, outputs=actor_output)
#
#     # ======================== CRITIC NETWORK =========================
#     critic_state_input = Input(shape=(1,) + env.observation_space.shape)  # State input
#     critic_action_input = Input(shape=(nb_actions,))  # Action input
#
#     x1 = Flatten()(critic_state_input)
#     x1 = Dense(256, activation="relu")(x1)
#     x1 = Dense(128, activation="relu")(x1)
#     x1 = Dense(64, activation="relu")(x1)
#
#     x2 = Dense(128, activation="relu")(critic_action_input)
#
#     merged = Concatenate()([x1, x2])  # Merge state & action
#     x = Dense(256, activation="relu")(merged)
#     x = Dense(128, activation="relu")(x)
#     x = Dense(64, activation="relu")(x)
#     critic_output = Dense(1, activation="linear")(x)  # Outputs Q-value
#     critic = Model(inputs=[critic_state_input, critic_action_input], outputs=critic_output)
#
#     # ===================== DDPG Agent Configuration =====================
#     memory = SequentialMemory(limit=50000, window_length=1)
#     random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.15, mu=0, sigma=0.2)
#
#     agent = DDPGAgent(actor=actor, critic=critic, nb_actions=nb_actions,
#                       memory=memory, nb_steps_warmup_critic=10, nb_steps_warmup_actor=10,
#                       random_process=random_process, gamma=0.99, target_model_update=1e-2,
#                       critic_action_input=critic_action_input, batch_size=16)  # ✅ FIXED
#
#     #agent.load_weights('ddpg_hexapod_weights.h5f')
#
#     agent.compile(Adam(learning_rate=0.1), metrics=['mae', 'mse'])
#
#     agent.fit(env, nb_steps=5000, visualize=False, verbose=1)  # Train for 50k steps
#
#     agent.save_weights('ddpg_hexapod_weights.h5f', overwrite=True)
#
# if __name__ == '__main__':
#     #print("Hi")
#     main()



# import time
#
# import numpy as np
# from pyrep import PyRep
# from pyrep.objects.joint import Joint
# from pyrep.objects.object import Object
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten
# tf.keras.__version__ = "2.11.0"
# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory
#
# import gym
# from tensorflow.keras.optimizers.legacy import Adam
# from HexapodEnv import HexapodEnv
#
#
# def working():
#     pr = PyRep()
#     # Launch the application with a scene file in headless mode
#     pr.launch('seminarka_scena.ttt', headless=True)
#     pr.start()  # Start the simulation
#
#     joint_names = [
#         f'hexa_joint{i}_{j}' for i in range(1, 4) for j in range(0,6)
#     ]
#     joints = [Joint(name) for name in joint_names]
#
#
#
#     # Define action space (joint angle limits)
#     action_range = {
#         'hip': (-0.5,0.5),
#         'knee': (-0.5,0.5),
#         'ankle': (-0.5,0.5)
#     }
#
#     def random_action():
#         """Generate random joint angles within the allowed range."""
#         return {
#             'hip': np.random.uniform(*action_range['hip']),
#             'knee': np.random.uniform(*action_range['knee']),
#             'ankle': np.random.uniform(*action_range['ankle'])
#         }
#
#     try:
#         for step in range(500):  # Run for 500 steps
#             pr.step()
#             action = random_action()
#
#             # Apply random movements to all 6 legs
#             for i in range(6):  # Loop through legs
#                 joints[i * 3].set_joint_target_position(action['hip'])
#                 joints[i * 3 + 1].set_joint_target_position(action['knee'])
#                 joints[i * 3 + 2].set_joint_target_position(action['ankle'])
#
#
#
#     finally:
#         pr.stop()
#         pr.shutdown()
#
# def main():
#     env = HexapodEnv()
#
#     # Create a simple neural network for RL
#     model = Sequential([
#         Flatten(input_shape=(1,) + env.observation_space.shape),
#         Dense(64, activation="relu"),
#         Dense(64, activation="relu"),
#         Dense(env.action_space, activation="linear")
#     ])
#
#     # Configure DQN agent
#     memory = SequentialMemory(limit=50000, window_length=1)
#     policy = EpsGreedyQPolicy()
#     dqn = DQNAgent(model=model, nb_actions=env.action_space, memory=memory,
#                    nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
#     dqn.compile(Adam(learning_rate=0.001), metrics=['mae'])
#
#     dqn.fit(env, nb_steps=500, visualize=True, verbose=1)
#
#     dqn.save_weights('dqn_hexapod_weights.h5f', overwrite=True)
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     #working()
#     main()
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
