import io
import os



import random
from collections import deque, defaultdict
import time
from contextlib import redirect_stdout
from datetime import datetime
from queue import Full, Empty


import cv2
import numpy as np

from DDPG.ou_noise import Noise
from DDPG.td3 import TD3
import tensorflow as tf

from pyrep.errors import PyRepError



class Worker:
    def __init__(self, directory_name, exp_queue, weights_queue, control_queue, results_queue, worker_id, env, td3: TD3, noise: Noise, load_weights=False):
        self.directory_name = directory_name
        self.env = env
        self.td3 = td3
        self.noise = noise
        # self.avg_episodes_rewards = []


        self.best_reward = float('-inf')
        self.best_actor = None
        self.best_critic_1 = None
        self.best_critic_2 = None

        self.training_logs = []
        self.config_logs = []

        self.max_episodes = 0
        self.max_steps = 0
        self.warm_up_memory = 0

        self.video_dir = 'recorded_episodes'
        self.all_recordings = []

        self.global_total_steps = 0

        self.exp_queue = exp_queue
        self.weights_queue = weights_queue
        self.control_queue = control_queue
        self.results_queue = results_queue
        self.worker_id = worker_id

        if self.worker_id == 0:
            self.manage_directory()

    """ Helper method, used to clear the current recording folder"""

    def _get_configuration(self):
        config_dict = defaultdict(int)
        # config_dict["Theta"] = self.noise.theta
        # config_dict["Dt"] = self.noise.dt
        # config_dict["Decay constant"] = self.noise.decay_constant
        config_dict["Max episodes"] = self.max_episodes
        config_dict["Max steps per episode"] = self.max_steps
        config_dict["Warm up steps"] = self.warm_up_memory


        return [f"{k}: {v}" for k, v in config_dict.items()]

    def manage_directory(self):
        os.makedirs(self.directory_name, exist_ok=True)

        for filename in os.listdir(self.directory_name):
            file_path = os.path.join(self.directory_name, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def update_normalizer_settings(self, normalizer_settings):
        self.td3.running_normalizer.adjust_settings(normalizer_settings)

    def update_min_max_normalizer_settings(self, min_max_normalizer_settings):
        self.env.velocity_normalizer.maximum_value = min_max_normalizer_settings


    def run_episodes(self, max_episodes=100, max_steps=500, warm_up_memory=500, preload_warmup=False):
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.warm_up_memory = warm_up_memory

        episodes_rewards = deque(maxlen=1000)
        successive_goals_reached = 0
        init_time = datetime.now()

        threshold_avg_rewards = [float('-inf')]
        walking_to_target = False
        # print(f"Min: {self.env.reward_min} | Max: {self.env.reward_max}")


        """ Warmup"""
        self.results_queue.put(f"Warm worker {self.worker_id} up for {warm_up_memory} steps")
        while self.env.current_global_step_count <= warm_up_memory:
            self.warmup_phase(None, max_steps=max_steps)

        self.env.current_global_step_count = 0
        self.results_queue.put(f"Warm up of {self.worker_id} finished")
        self.control_queue.put(f"WARMUP FINISHED")

        """ Send velocity"""
        self.control_queue.put(self.env.velocity_normalizer.maximum_value)

        """ Wait for normalizer settings """
        normalizer_settings, min_max_velocity_normalizer_settings = self.weights_queue.get()

        self.update_normalizer_settings(normalizer_settings)
        self.update_min_max_normalizer_settings(min_max_velocity_normalizer_settings)

        self.results_queue.put(f"Normalizer of {self.worker_id} settings updated: {normalizer_settings} | velocity: {min_max_velocity_normalizer_settings}")

        # sleep_time = random.uniform(1, 60)  # float (e.g., 37.42 seconds)
        # time.sleep(sleep_time)


        for episode in range(max_episodes + 1):
            # if self.env.current_global_step_count >= self.env.phase_limit:
            #     if self.env.current_phase < self.env.max_phase_num:
            #         self.env.current_phase += 1
            #
            #         self.results_queue.put(f"Moving worker {self.worker_id} to the next phase, current phase is: {self.env.current_phase}")
            #         episodes_rewards.clear()
            #         self.control_queue.put("CLEAR")
            #
            #         self.env.current_global_step_count = 0
            #         self.results_queue.put(f"Warm worker {self.worker_id} up for {warm_up_memory} steps")
            #         while self.env.current_global_step_count <= warm_up_memory:
            #             self.warmup_phase(None, max_steps=max_steps)
            #         self.results_queue.put(f"Warm up of {self.worker_id} finished")
            #         self.control_queue.put(f"WARMUP FINISHED")
            #         time.sleep(0.5)
            #
            #         if self.env.current_phase == 3:
            #             self.max_steps = 200

            start_time = time.time()
            current_avg = 0.0 if len(episodes_rewards) == 0 else float(np.mean(episodes_rewards))

            try:
                msg = self.control_queue.get_nowait()
                #self.results_queue.put(f"Original message: {msg}")
                if msg == "STOP WORKER":
                    #self.results_queue.put("WORKER WANTS TO STOP")
                    break
            except Empty:
                pass

            reward_episode, goal_reached, steps_taken = self.run_episode(episode, init_time, current_avg)

            end_time = time.time()

            if goal_reached:
                successive_goals_reached += 1
                self.training_logs.append("Target reached")
            else:
                successive_goals_reached = 0

            episodes_rewards.append(reward_episode)
            avg_reward = round(np.mean(episodes_rewards), 5)

            self.results_queue.put_nowait(
                f"W: {self.worker_id}; Ep * {episode} * - R: {reward_episode} ; Avg current ep => {round(reward_episode / steps_taken, 2)}; Avg per 1000 => {avg_reward} ; Noise: {self.noise.std_dev[0]} ; Time ==> {round(end_time - start_time, 2)}")
            self.training_logs.append(
                f"Ep * {episode} * - R: {reward_episode} ; Avg current ep => {round(reward_episode / steps_taken, 2)}; Avg per 1000 => {avg_reward} ; Noise: {self.noise.std_dev[0]} ; Time ==> {round(end_time - start_time, 2)}")

            if self.worker_id == 0:
                """ Tensorboard logging logic """
                with self.td3.summary_writer.as_default():
                    tf.summary.scalar('Episode Reward', reward_episode, step=episode)
                    tf.summary.scalar('Avg Step Reward', reward_episode / steps_taken, step=episode)

            if successive_goals_reached >= 100 and not walking_to_target:
                message = "Agent is walking to target confidently"
                self.training_logs.append(message)
                self.results_queue.put(message)
                walking_to_target = True
                #break
            avg_reward_small = round(np.mean(list(episodes_rewards)[-200:]), 3)

            if episode % 200 == 0 and episode > 0:
                # Noise decay
                # self.noise.decay(self.env.action_space.shape)
                #
                # message = f"Noise decayed to {self.noise.std_dev[0]}"
                # self.training_logs.append(message)
                # self.results_queue.put(message)

                if avg_reward < threshold_avg_rewards[-1] and walking_to_target:
                    message = "Agent learned policy"
                    self.training_logs.append(message)
                    self.results_queue.put(message)
                    break
                threshold_avg_rewards.append(avg_reward)

        self.results_queue.put(f"Worker {self.worker_id} finished")

        self.control_queue.put("STOP TEACHER")


        if self.worker_id == 0:
            config = self._get_configuration()
            with open(f"{os.path.join(self.directory_name, 'config_logs_worker.txt')}", "w") as f:
                f.write("\n".join(config))

            with open(f"{os.path.join(self.directory_name, 'training_logs.txt')}", "w") as f:
                f.write("\n".join(self.training_logs))

        #self.results_queue.put(f"Worker {self.worker_id} finished")
        self.env.close()
        os._exit(0)
    """ Main simulation logic """

    def run_episode(self, episode, start_time=time.time(), avg_reward=0.0):
        try:
            state = self.env.reset()
            self.noise.reset_noise()

            reward_from_episode = 0
            current_steps = 0
            frames = []
            goal_reached = False

            """ Update the networks from queue """
            actor_weights = None

            drained = 0
            try:
                while True:  # drain queue, keep newest
                    actor_weights = self.weights_queue.get_nowait()
                    drained += 1
            except Empty:
                if actor_weights is not None:
                    self.td3.actor.set_weights(actor_weights)

                pass

            #self.results_queue.put(f"W {self.worker_id} | {drained} | {self.weights_queue.qsize()}")

            episode_states = []
            previous_action = np.zeros_like(self.env.action_space.shape)
            while current_steps < self.max_steps:
                self.env.current_global_step_count += 1
                actions = self._policy(state, self.noise, episode, previous_action)
                previous_action = actions

                """ Joint target positions set """
                self.env.set_joint_positions(actions, episode)

                """ Step the simulation """
                self.env.step_and_update()
                self.env.current_step += 1

                if episode % 500 == 0 and self.worker_id == 0:
                    """ We use handle_explicitly to turn on the sensor for a this capture"""
                    self.env.vision_sensor.handle_explicitly()
                    img = self.env.vision_sensor.capture_rgb()
                    img = (img * 255).astype(np.uint8)
                    frames.append(img)

                current_simulation_count = 0
                """ Wait until physic simulation is synced"""
                while not self.env.all_joints_reached_targets():
                    current_simulation_count += 1

                    if current_simulation_count > self.env.max_simulation_count:
                        break

                    """ Step the simulation """
                    self.env.step_and_update()

                    if episode % 500 == 0 and self.worker_id == 0:
                        """ We use handle_explicitly to turn on the sensor for a this capture"""
                        self.env.vision_sensor.handle_explicitly()
                        img = self.env.vision_sensor.capture_rgb()
                        img = (img * 255).astype(np.uint8)
                        frames.append(img)

                next_state, reward, done, goal_reached = self.env.get_step_results(episode)


                episode_states.append(state)

                try:
                    self.exp_queue.put_nowait((state, actions, reward, next_state, done))
                except Full:
                    self.results_queue.put_nowait(f"EXP QUEUE is full")
                    pass

                reward_from_episode += reward

                current_steps += 1

                if done or goal_reached:
                    break

                state = next_state

            """ Update running normalizer after episode ends"""
            #self._update_observation_space(episode_states)

            # print(self.ddpg.running_normalizer.mean)
            # print(episode_states[0])
            # print(self.ddpg.running_normalizer.normalize(episode_states[0]))
            # print("---------")

            reward_from_episode = round(reward_from_episode, 5)
            if episode % 500 == 0 and self.worker_id == 0:
                self._record_current_frames(episode, frames, start_time, reward_from_episode, avg_reward)

            return reward_from_episode, goal_reached, current_steps
        except Exception as e:
            self.results_queue.put(f"ERROR. Worker {self.worker_id} encountered {e}")

    def warmup_phase(self, episode, max_steps=500):
        state = self.env.reset()
        #self.noise.reset_noise()

        current_steps = 0

        episode_states = []
        previous_action = np.zeros_like(self.env.action_space.shape)
        while current_steps < max_steps:
            try:
                self.env.current_global_step_count += 1
                actions = self._policy(state, self.noise, episode, previous_action, warmup_phase=True)
                previous_action = actions

                """ Joint target positions set """
                #TODO UNCOMMENT
                self.env.set_joint_positions(actions, episode)

                """ Step the simulation """
                self.env.step_and_update()

                current_simulation_count = 0
                """ Wait until physic simulation is synced"""
                while not self.env.all_joints_reached_targets():
                    current_simulation_count += 1
                    # joint_target_pos = []
                    # joint_current_pos = []
                    # joint_velocity = []
                    #
                    # self.results_queue.put(f"Current sim: {current_simulation_count}")
                    # for joint in self.env.joints:
                    #     joint_current_pos.append(joint.get_joint_position())
                    #     joint_target_pos.append(joint.get_joint_target_position())
                    #     joint_velocity.append(joint.get_joint_velocity())
                    #     break
                    #
                    # self.results_queue.put(f"INFO  {joint_current_pos} | {joint_target_pos} | {joint_velocity}")

                    # joint_target_pos = []
                    # joint_current_pos = []
                    # joint_velocity = []
                    # if current_simulation_count == 1:
                    #     for joint in self.env.joints:
                    #         joint_current_pos.append(joint.get_joint_position())
                    #         joint_target_pos.append(joint.get_joint_target_position())
                    #         joint_velocity.append(joint.get_joint_velocity())
                    #     self.results_queue.put(f"INFO  {joint_current_pos} | {joint_target_pos} | {joint_velocity}")

                    if current_simulation_count > self.env.max_simulation_count:
                        # joint_target_pos = []
                        # joint_current_pos = []
                        # joint_velocity = []
                        # for joint in self.env.joints:
                        #     joint_current_pos.append(joint.get_joint_position())
                        #     joint_target_pos.append(joint.get_joint_target_position())
                        #     joint_velocity.append(joint.get_joint_velocity())
                        # self.results_queue.put(f"Joints not synced {joint_current_pos} | {joint_target_pos} | {joint_velocity}")
                        #self.results_queue.put(f"Joints not synced")
                        break

                    """ Step the simulation """
                    self.env.step_and_update()

                next_state, reward, done, goal_reached= self.env.get_step_results(episode, warmup = True)
                #self.results_queue.put(f"R: {leg_rewards}")
                #tip_distances = next_state[-7:-1]
                episode_states.append(state)

                try:
                    self.exp_queue.put_nowait((state, actions, reward, next_state, done))
                except Full:
                    self.results_queue.put_nowait(f"EXP QUEUE is full")
                    pass

                current_steps += 1
                if done or goal_reached:
                    break
                state = next_state
            except PyRepError as e:
                self.results_queue.put(f"Coppeliasim error occured {e}")
            except RuntimeError as e:
                self.results_queue.put(f"Coppeliasim runtime error occured {e}")
        self._update_observation_space(episode_states)

    # def _forward(self, state):
    #     state = np.expand_dims(state, axis=0)
    #     action = self.td3.actor(state)
    #     return np.squeeze(action, axis=0)
    @tf.function
    def _forward(self, state):
        return tf.squeeze(self.td3.actor(tf.expand_dims(state, 0)), axis=0)

    """ Method responsible returning state based on current action and added noise"""
    def _policy(self, state, noise, episode, previous_action, warmup_phase=False, training = True):
        if warmup_phase:
            # Generate random joint position for each joint - in warmup phase
            legal_action = np.array([
                random.uniform(self.td3.lower_bound, self.td3.upper_bound)
                #random.uniform(0.0, 0.0)
                for _ in range(self.td3.number_of_actions[0])
            ], dtype=np.float32)

            #sampled_actions = 0.9 * previous_action + 0.1 * legal_action
            sampled_actions = legal_action

            legal_action = np.clip(sampled_actions, self.td3.lower_bound, self.td3.upper_bound)

        else:
            normalized_state = self._normalize_observation_space(state)
            #
            sampled_actions = self._forward(normalized_state)
            # sampled_actions = np.array([
            #     random.uniform(self.td3.lower_bound, self.td3.upper_bound)
            #     # random.uniform(0.0, 0.0)
            #     for _ in range(self.td3.number_of_actions[0])
            # ], dtype=np.float32)

            if training:
                #noise_o = noise()

                # policy_noise = 0.3 * 1.5  # ≈ 0.45
                # noise_clip = 0.5 * 1.5  # ≈ 0.75
                # noise = np.random.normal(0, policy_noise, size=self.env.action_space.shape[0])
                # noise = np.clip(noise, -noise_clip, noise_clip)

                noise = noise()

                new_actions = sampled_actions + noise

                #sampled_actions = 0.2 * previous_action + 0.8 * new_actions
                sampled_actions = new_actions

                self.noise.decay()

            legal_action = np.clip(sampled_actions, self.td3.lower_bound, self.td3.upper_bound)


        return legal_action

    """ Method responsible for normalizing bathc of states"""
    def _normalize_observation_space(self, current_state):
        current_state = np.asarray(current_state, dtype=np.float32)

        if current_state.ndim == 1:
            return np.array(self.td3.running_normalizer.normalize(current_state))
        else:
            states = []
            for state in current_state:
                states.append(self.td3.running_normalizer.normalize(state))

            return np.array(states)

    """ Method responsible for updating normalizer based on current state"""
    def _update_observation_space(self, episode_states):
        for state in episode_states:
            self.td3.running_normalizer.update(state)

    def _record_current_frames(self, episode, frames, start_time, reward_from_episode, avg_reward):
        os.makedirs(self.directory_name, exist_ok=True)

        #video_path = os.path.join(self.directory_name, f"episode_{episode}.avi")
        video_path = os.path.join(self.directory_name, f"episode_{episode}.mp4")

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 0, 255)
        thickness = 1
        position = (10, 30)

        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            now_time = datetime.now() - start_time
            hours = now_time.days * 24 + now_time.seconds // 3600
            minutes = (now_time.seconds % 3600) // 60
            cv2.putText(bgr_frame, f'Episode: {episode}, training time: {hours} hours, {minutes} minutes | Episode reward: {round(reward_from_episode, 2)} | Average Episode reward: {round(avg_reward, 2)}', position, font, font_scale, font_color, thickness,
                        cv2.LINE_AA)
            out.write(bgr_frame)

        out.release()
        self.all_recordings.append(video_path)