import io
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from collections import defaultdict
from contextlib import redirect_stdout
from queue import Empty, Full

import numpy as np

#from DDPG.her_memory import HerMemory
from DDPG.memory import Memory
from DDPG.td3 import TD3
import tensorflow as tf

from DDPG.td3_logic import TD3Logic


class Teacher:
    def __init__(self, exp_queue, weight_queues, control_queues, env, td3: TD3Logic, memory: Memory, current_num_of_workers, directory_name):
        self.exp_queue = exp_queue
        self.weight_queues = weight_queues
        self.control_queues = control_queues
        self.env = env
        self.td3 = td3
        self.memory = memory
        self.global_total_steps = 0
        self.current_num_of_workers = current_num_of_workers
        self.directory_name = directory_name


    def _get_configuration(self):
        config_dict = defaultdict(int)
        config_dict["Learning rate actor"] = self.td3.learning_rate_actor
        config_dict["Learning rate critic"] = self.td3.learning_rate_critic
        config_dict["Gamma"] = self.td3.gamma
        config_dict["Tau"] = self.td3.tau
        config_dict["Memory capacity"] = self.memory.max_capacity
        config_dict["Batch size"] = self.memory.batch_size
        # config_dict["Theta"] = self.noise.theta
        # config_dict["Dt"] = self.noise.dt
        # config_dict["Decay constant"] = self.noise.decay_constant
        # config_dict["Max episodes"] = self.max_episodes
        # config_dict["Max steps"] = self.max_steps
        # config_dict["Warm up steps"] = self.warm_up_memory

        stream = io.StringIO()
        with redirect_stdout(stream):
            self.td3.actor.summary()
        summary_str = stream.getvalue()

        config_dict["Actor architecture"] = summary_str

        stream = io.StringIO()
        with redirect_stdout(stream):
            self.td3.critic_1.summary()
        summary_str = stream.getvalue()
        config_dict["Critic architecture - using two of them"] = summary_str


        return [f"{k}: {v}" for k, v in config_dict.items()]

    def update_network(self):
        """ Update the networks every X steps"""

        self.global_total_steps += 1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample_from_memory()

        state_batch = self._normalize_observation_space(state_batch)
        next_state_batch = self._normalize_observation_space(next_state_batch)


        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            next_state_batch, dtype=tf.float32
        )
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.bool)


        self.td3.train(self.global_total_steps, state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def update_workers(self, actor_weights):
        for queue_id, worker_weight_queue in enumerate(self.weight_queues):
            try:
                worker_weight_queue.put_nowait(actor_weights)
            except Full:
                #print(f"WEIGHT QUEUE is full: {queue_id}")
                pass

    def _update_observation_space(self, transition):
        state, actions, reward, next_state, done = transition
        self.td3.running_normalizer.update(state)
        self.td3.running_normalizer.update(next_state)

    def _normalize_observation_space(self, batch_states):
        states = []
        for state in batch_states:
            states.append(self.td3.running_normalizer.normalize(state))

        return np.array(states)


    def teacher_loop(self):
        """ IN seconds"""
        SYNC_INTERVAL = 10.0
        last_sync = time.time()

        new_experiences = 0




        terminate = False
        run = True
        warmup = True
        maximum_velocity_value = float('-inf')
        try:
            while not terminate:
                warmup_worker_finished = 0

                while warmup:
                    try:
                        while True:
                            transition = self.exp_queue.get_nowait()
                            self.memory.insert_to_memory(transition)
                            self._update_observation_space(transition)


                            #print(f"{self.memory.current_capacity} || {id}")
                    except Empty:
                        pass

                    for queue in self.control_queues:
                        try:
                            msg = queue.get_nowait()
                            if msg == "WARMUP FINISHED":
                                warmup_worker_finished += 1
                                velocity_value = queue.get()

                                maximum_velocity_value = max(velocity_value, maximum_velocity_value)


                            if warmup_worker_finished == self.current_num_of_workers:
                                warmup = False

                                normalizer_settings = [self.td3.running_normalizer.mean, self.td3.running_normalizer.M2,
                                                       self.td3.running_normalizer.count,
                                                       self.td3.running_normalizer.epsilon]

                                for c_queue in self.weight_queues:
                                    c_queue.put((normalizer_settings, maximum_velocity_value))
                                break
                        except Empty:
                            continue

                while run:
                    try:
                        while True:
                            transition = self.exp_queue.get_nowait()
                            self.memory.insert_to_memory(transition)
                            new_experiences += 1
                            #self._update_observation_space(transition)

                            # if self.memory.current_capacity >= self.memory.batch_size:
                            #     trans_count += 1
                    except Empty:
                        pass
                    #print(self.memory.current_capacity)
                    # if new_experiences >= 100 and self.memory.current_capacity < self.memory.max_capacity:
                    #     self.update_network()
                    #     new_experiences = 0
                    # elif new_experiences >= 1 and self.memory.current_capacity >= self.memory.max_capacity:
                    #     self.update_network()
                    #     new_experiences = 0

                    if new_experiences >= 50:
                        self.update_network()
                        new_experiences = 0


                    if time.time() - last_sync > SYNC_INTERVAL and self.memory.current_capacity >= self.memory.batch_size:
                        # print("Trans count: ", trans_count)
                        # trans_count = 0
                        # print("Sending netw")

                        actor_weights = self.td3.actor.get_weights()
                        #critic_1_weights, critic_2_weights = self.td3.critic_1.get_weights(), self.td3.critic_2.get_weights()
                        self.update_workers(actor_weights)
                        last_sync = time.time()
                        #print("sending netw")

                    for id, queue in enumerate(self.control_queues):
                        try:
                            msg = queue.get_nowait()
                        except Empty:
                            continue

                        if msg == "STOP TEACHER":
                            run = False
                            terminate = True
                            break
                        elif msg == "CLEAR":
                            self.memory.clear_memory()
                            run = False
                        else:
                            print(f"Message: {msg}")

        except KeyboardInterrupt:
            print("INTERRUPT")
            # for control_queue in self.control_queues:
            #     control_queue.put("STOP WORKER")

        print("STOPPING WORKERS")
        for control_queue in self.control_queues:
            try:
                for _ in range(11):
                    control_queue.get_nowait()
            except Empty:
                control_queue.put_nowait("STOP WORKER")

        config = self._get_configuration()
        with open(f"{os.path.join(self.directory_name, 'config_logs_teacher.txt')}", "w") as f:
            f.write("\n".join(config))

        self.td3.actor.save_weights(os.path.join(self.directory_name, "hexapod_actor.h5"))
        self.td3.critic_1.save_weights(os.path.join(self.directory_name, "hexapod_critic_1.h5"))
        self.td3.critic_2.save_weights(os.path.join(self.directory_name, "hexapod_critic_2.h5"))

        print("Teacher finished")
