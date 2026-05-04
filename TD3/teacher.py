import io
import os
import pickle
import time
from collections import defaultdict
from contextlib import redirect_stdout
from queue import Empty, Full
import gc
import numpy as np
from TD3.memory import Memory
import tensorflow as tf
from TD3.td3_logic import TD3Logic

class Teacher:
    def __init__(self, exp_queue, weight_queues, control_queues, td3: TD3Logic, memory: Memory, current_num_of_workers, directory_name, update_rate):
        """
        Initialization of the teacher process
        :param exp_queue: experience queue from which the teacher gathers feedback from each worker's interactions
        :param weight_queues: a queues used for sending each worker new actor weights
        :param control_queues: synchronization queues used between worker and teacher
        :param td3: instance of TD3Logic class which contain actor and critic's networks
        :param memory: instance of Memory class implementing replay buffer logic
        :param current_num_of_workers: chosen number of worker processes
        :param directory_name: ame of the directory in which the results will be saved
        :param update_rate: chosen network update rate
        """
        self.exp_queue = exp_queue
        self.weight_queues = weight_queues
        self.control_queues = control_queues
        self.td3 = td3

        self.memory = memory
        self.num_of_network_updates = 0
        self.current_num_of_workers = current_num_of_workers
        self.directory_name = directory_name
        self.best_critic1 = None
        self.best_critic2 = None
        self.update_rate = update_rate
        self.td3.log_interval = max(1, int(self.td3.base_log_interval / (1 + update_rate)))

    def _get_configuration(self):
        """
        Method responsible for collecting some of the training settings
        :return: key-value pairs of said settings
        """
        config_dict = defaultdict(int)
        config_dict["Learning rate actor"] = self.td3.learning_rate_actor
        config_dict["Learning rate critic"] = self.td3.learning_rate_critic
        config_dict["Gamma"] = self.td3.gamma
        config_dict["Tau"] = self.td3.tau
        config_dict["Memory capacity"] = self.memory.max_capacity
        config_dict["Batch size"] = self.memory.batch_size

        stream = io.StringIO()
        with redirect_stdout(stream):
            self.td3.actor.summary()
        summary_str = stream.getvalue()
        config_dict["Actor architecture"] = summary_str

        stream = io.StringIO()
        with redirect_stdout(stream):
            self.td3.critic_1.summary()
        summary_str = stream.getvalue()
        config_dict["Critic architecture"] = summary_str

        return [f"{k}: {v}" for k, v in config_dict.items()]

    def _update_network(self):
        """
        Method responsible for sampling entries from replay buffer, normalizing them and then updating the networks
        :return:
        """
        self.num_of_network_updates += 1
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

        self.td3.train(self.num_of_network_updates, state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def _update_workers(self, actor_weights):
        """
        Method responsible for sending new actor weights to each worker's queue
        :param actor_weights: current actor weights
        :return:
        """
        """ Drain queues first """
        for w_queue in self.weight_queues:
            try:
                while True:
                    _ = w_queue.get_nowait()
            except Empty:
                pass

        for queue_id, worker_weight_queue in enumerate(self.weight_queues):
            try:
                worker_weight_queue.put_nowait(actor_weights)
            except Full:
                print(f"WEIGHT QUEUE is full: {queue_id}")
                pass

    def _update_observation_space(self, transition):
        """
        Method used to update state normalizer settings -- during the warmup phase
        :param transition:
        :return:
        """
        state, actions, reward, next_state, done = transition
        self.td3.running_normalizer.update(state)
        if done:
            self.td3.running_normalizer.update(next_state)

    def _normalize_observation_space(self, batch_states):
        """
        Method used to normalize sampled batch of states from replay buffer
        :param batch_states: sampled batch of states
        :return:
        """
        states = []
        for state in batch_states:
            states.append(self.td3.running_normalizer.normalize(state))

        return np.array(states)

    def _teacher_warmup_logic(self):
        """
        Main simulation loop on teacher's side - logic containing warmup phase
        :return:
        """
        warmup_worker_finished = 0
        warmup = True
        maximum_velocity_value = float('-inf')

        while warmup:
            try:
                while True:
                    transition = self.exp_queue.get_nowait()
                    self.memory.insert_to_memory(transition)
                    self._update_observation_space(transition)
            except Empty:
                pass

            for queue in self.control_queues:
                try:
                    msg = queue.get_nowait()
                    if msg == "WARMUP FINISHED":
                        warmup_worker_finished += 1
                        velocity_value = queue.get()
                        maximum_velocity_value = max(velocity_value, maximum_velocity_value)
                    else:
                        try:
                            queue.put_nowait(msg)
                        except Full:
                            pass
                except Empty:
                    continue

            if warmup_worker_finished == self.current_num_of_workers:
                normalizer_settings = [self.td3.running_normalizer.mean, self.td3.running_normalizer.M2,
                                       self.td3.running_normalizer.count,
                                       self.td3.running_normalizer.epsilon]

                with open(os.path.join(self.directory_name, "running_normalizer_data.pkl"), "wb") as f:
                    pickle.dump(normalizer_settings, f)

                with open(os.path.join(self.directory_name, "min_max_normalizer_data.pkl"), "wb") as f:
                    pickle.dump(maximum_velocity_value, f)

                """ Drain queues first """
                try:
                    for w_queue in self.weight_queues:
                        try:
                            while True:
                                _ = w_queue.get_nowait()
                        except Empty:
                            pass
                except Empty:
                    pass

                """ Send final normalizer's settings to each worker """
                for c_queue in self.weight_queues:
                    c_queue.put((normalizer_settings, maximum_velocity_value))

                if self.best_critic1 is None:
                    self.best_critic1 = tf.keras.models.clone_model(self.td3.critic_1)
                    self.best_critic2 = tf.keras.models.clone_model(self.td3.critic_2)
                self.best_critic1.set_weights(self.td3.critic_1.get_weights())
                self.best_critic2.set_weights(self.td3.critic_2.get_weights())

                break

    def _cleanup_logic(self):
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

        self.best_critic1.save_weights(
            os.path.join(self.directory_name, f"hexapod_critic1.h5"))
        self.best_critic2.save_weights(
            os.path.join(self.directory_name, f"hexapod_critic2.h5"))
    def teacher_loop(self):
        """
        Main simulation loop on teacher's side - logic containing training phase
        :return:
        """
        SYNC_INTERVAL = 10.0
        last_sync = time.time()
        new_experiences = 0

        terminate = False
        try:
            while not terminate:
                run = True
                self._teacher_warmup_logic()

                while run:
                    try:
                        while True:
                            transition = self.exp_queue.get_nowait()
                            self.memory.insert_to_memory(transition)
                            new_experiences += 1
                    except Empty:
                        pass

                    if new_experiences >= self.update_rate:
                        self._update_network()
                        new_experiences = 0

                    if time.time() - last_sync > SYNC_INTERVAL and self.memory.current_capacity >= self.memory.batch_size:
                        actor_weights = self.td3.actor.get_weights()
                        self._update_workers(actor_weights)
                        last_sync = time.time()

                        gc.collect()

                    control_messages = []
                    for queue in self.control_queues:
                        try:
                            msg = queue.get_nowait()
                            control_messages.append(msg)
                        except Empty:
                            continue

                    if "STOP TEACHER" in control_messages:
                        terminate = True
                        break
                    elif "SAVE" in control_messages:
                        print("Teacher is updating best critic networks")
                        if self.best_critic1 is None:
                            self.best_critic1 = tf.keras.models.clone_model(self.td3.critic_1)
                            self.best_critic2 = tf.keras.models.clone_model(self.td3.critic_2)
                        self.best_critic1.set_weights(self.td3.critic_1.get_weights())
                        self.best_critic2.set_weights(self.td3.critic_2.get_weights())

                        self.best_critic1.save_weights(
                            os.path.join(self.directory_name, f"hexapod_critic1.h5"))
                        self.best_critic2.save_weights(
                            os.path.join(self.directory_name, f"hexapod_critic2.h5"))
                    else:
                        if len(control_messages) != 0:
                            print(f"Message: {control_messages}")

        except Exception as e :
            print(f"Error occured: {e}")

        self._cleanup_logic()
        print("Teacher finished")
