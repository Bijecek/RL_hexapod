import os.path
import shutil
import threading
from queue import Empty

import numpy as np
import sys

from tensorflow.python.ops.gen_experimental_dataset_ops import experimental_csv_dataset

from DDPG.agent import Agent
from DDPG.ou_noise import Noise
# from DDPG.ddpg import DDPG
from DDPG.td3 import TD3
from DDPG.memory import Memory
from DDPG.noise import GaussianNoise
from DDPG.td3_logic import TD3Logic
from DDPG.teacher import Teacher
from DDPG.worker import Worker
from hexapod_env import HexapodEnv
import multiprocessing as mp
import time
import threading



#
# def _run(env, directory_name):
#     td3 = TD3(env.observation_space.shape, env.action_space.shape,
#                 env.action_space.high[0], env.action_space.low[0],
#                 0.0001, 0.001, gamma=0.99, tau=0.005)
#     memory = Memory(1_000_000, env.observation_space.shape[0], env.action_space.shape[0], batch_size=256)
#
#
#     noise = Noise(std_deviation=float(0.3),
#                   theta=0.15, dt=0.05, decay_constant=0.98, noise_shape=env.action_space.shape)
#
#     agent = Agent(env, td3, memory, noise, False, directory_name)
#     agent.run_episodes(max_episodes=20_000, max_steps=20, warm_up_memory=3_000, preload_warmup=False)
#     agent.save_results(directory_name)
#     agent.merge_recordings()

def _run_worker(directory_name, exp_queue, weights_queue, control_queue, results_queue, worker_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    env = HexapodEnv(results_queue, disable_rendering=True, rewards=2)
    td3 = TD3(env.observation_space.shape, env.action_space.shape,
                env.action_space.high[0], env.action_space.low[0])

    # noise = Noise(
    #     std_start= 0.6,
    #     std_end= 0.05,
    #     max_steps=15_000 * 50,
    #     std_deviation=None,
    #               theta=0.15, dt=1.0, decay_constant=None, noise_shape=env.action_space.shape)
    noise = Noise(
            std_start= 0.2,
            std_end= 0.05,
            max_steps=3000 * 150,
            std_deviation=None,
                      theta=0.15, dt=1.0, decay_constant=None, noise_shape=env.action_space.shape)


    # noise = GaussianNoise(
    #     std_start=0.6,
    #     std_end=0.075,
    #     max_steps=20_000 * 50,
    #     noise_shape=(18,),
    #     clip=0.75
    # )

    worker = Worker(directory_name, exp_queue, weights_queue, control_queue, results_queue, worker_id, env, td3, noise, False)
    worker.run_episodes(max_episodes=40_000, max_steps=150, warm_up_memory=10_000, preload_warmup=False)
    #worker.run_episodes(max_episodes=30, max_steps=100, warm_up_memory=100, preload_warmup=False)

def _run_teacher(exp_queue, weights_queues, control_queues, current_num_of_workers, directory_name):
    env = HexapodEnv(None, disable_rendering=True, rewards=2)
    # td3 = TD3Logic(env.observation_space.shape, env.action_space.shape,
    #           env.action_space.high[0], env.action_space.low[0],
    #           0.0001, 0.001, gamma=0.99, tau=0.005)
    td3 = TD3Logic(env.observation_space.shape, env.action_space.shape,
                   env.action_space.high[0], env.action_space.low[0],
                   0.0001, 0.001, gamma=0.99, tau=0.005)


    memory = Memory(200_000, env.observation_space.shape[0], env.action_space.shape[0], batch_size=256 )

    teacher = Teacher(exp_queue, weights_queues, control_queues, env, td3, memory, current_num_of_workers, directory_name)
    teacher.teacher_loop()

def _print_results(results_queues):
#    last_print = time.time()
    run = True

    while run:
        for q in results_queues:
            try:
                msg = q.get(timeout=0.1)
                if msg == "STOP printer":
                    print("STOPPING printer")
                    run = False
                    break
                else:
                    print(msg)
            except Empty:
                pass
            except Exception as e:
                run = False
                break


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

    mp.set_start_method("spawn", force=True)
    n_workers = 8
    print(f"Starting with {n_workers} workers")

    exp_queue = mp.Queue(maxsize=1_000)
    weights_queues = [mp.Queue(maxsize=10) for _ in range(n_workers)]
    control_queues = [mp.Queue(maxsize=10) for _ in range(n_workers)]
    results_queues = [mp.Queue(maxsize=100) for _ in range(n_workers)]


    workers = []
    for wid in range(n_workers):
        p = mp.Process(
            target=_run_worker,
            args=(directory_name, exp_queue, weights_queues[wid], control_queues[wid], results_queues[wid], wid)
        )
        p.start()
        workers.append(p)

    printer_thread = threading.Thread(
        target=_print_results,
        args=(results_queues,),
    )
    printer_thread.start()

    _run_teacher(exp_queue, weights_queues,control_queues, n_workers, directory_name)


    for id, p in enumerate(workers):
        print(f"Waiting for {id}")
        p.join()

    print("Workers finished")
    exp_queue.close()
    exp_queue.join_thread()

    #results_queues[0].put("STOP printer")


    print("EXP QUEUE CLOSED")
    for q in weights_queues:
        q.cancel_join_thread()
        q.close()


    print("WEIGHT CLOSED")

    for q in control_queues:
        q.close()
        q.join_thread()

    print("CONTROL CLOSED")


    for q in results_queues:
        q.close()

        q.join_thread()


    print("RESULT CLOSED")



def main():
    if len(sys.argv) == 3:
        """ Training """
        if sys.argv[2] == "Train":
            directory_name = sys.argv[1] if len(sys.argv) > 2 else None
            print(directory_name)
            experiment(directory_name, disable_rendering=False)
        # elif sys.argv[2] == "Test":
        #     """ Testing """
        #     experiment_test(directory_name=None, disable_rendering=False)

if __name__ == '__main__':
    main()