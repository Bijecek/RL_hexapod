import os
import pickle
import random
from collections import deque, defaultdict
import time
from datetime import datetime
from queue import Full, Empty
import subprocess
import cv2
import numpy as np
from TD3.ou_noise import Noise
from TD3.td3 import TD3
import tensorflow as tf
from pyrep.errors import PyRepError
from tensorflow.keras import backend as K

class Worker:
    def __init__(self, directory_name, exp_queue, weight_queue, control_queue, results_queue, worker_id, env, td3: TD3, noise):
        """
        Initialization of the worker class
        :param directory_name: name of the directory in which the results will be saved
        :param exp_queue: experience queue in which worker inserts feedback from env
        :param weight_queue: a queue used for receiving new actor weights
        :param control_queue: synchronization queue used between worker and teacher
        :param results_queue: results queue in which each worker inserts current episode performance
        :param worker_id: current worker identification number
        :param env: instance of the HexapodEnv class representing the environment
        :param td3: instance of the TD3 class containing actor's network
        :param noise: instance of OUNoise class
        """
        self.directory_name = directory_name
        self.env = env
        self.td3 = td3
        self.noise = noise
        self.best_reward = float('-inf')
        self.best_actor = None
        self.training_logs = []
        self.max_episodes = 0
        self.max_steps = 0
        self.warm_up_memory = 0
        self.video_dir = 'recorded_episodes'
        self.all_recordings = []

        self.exp_queue = exp_queue
        self.weight_queue = weight_queue
        self.control_queue = control_queue
        self.results_queue = results_queue
        self.worker_id = worker_id

        if self.worker_id == 0:
            self.manage_directory()

        random.seed(worker_id)
        np.random.seed(worker_id)

    def _get_configuration(self):
        """
        Method responsible for collecting some of the training settings
        :return: key-value pairs of said settings
        """
        config_dict = defaultdict(int)
        config_dict["Max episodes"] = self.max_episodes
        config_dict["Max steps per episode"] = self.max_steps
        config_dict["Warm up steps"] = self.warm_up_memory

        return [f"{k}: {v}" for k, v in config_dict.items()]

    def manage_directory(self):
        """
        Method responsible for directory cleanup
        :return:
        """
        os.makedirs(self.directory_name, exist_ok=True)

        for filename in os.listdir(self.directory_name):
            file_path = os.path.join(self.directory_name, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


    def update_normalizer_settings(self, normalizer_settings):
        """
        Method used for updating normalizer settings when teacher sends the final one
        :param normalizer_settings: final normalizer settings
        :return:
        """
        self.td3.running_normalizer.adjust_settings(normalizer_settings)

    def update_min_max_normalizer_settings(self, min_max_normalizer_settings):
        """
        Method used for updating min-max normalizer settings when teacher sends the final one
        :param min_max_normalizer_settings: final normalizer settings
        :return:
        """
        self.env.velocity_normalizer.maximum_value = min_max_normalizer_settings


    def run_episodes(self, max_episodes=100, max_steps=500, warm_up_memory=500):
        """
        Method containing main simulation loop on worker's side
        :param max_episodes: Constant determining maximum possible episodes
        :param max_steps: Constant determining maximum steps per episode
        :param warm_up_memory: Constant determining number of collected transition samples before training begins
        :return:
        """
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.warm_up_memory = warm_up_memory

        episodes_rewards = deque(maxlen=1000)
        successive_goals_reached = 0
        init_time = datetime.now()
        threshold_avg_rewards = float('-inf')
        walking_to_target = False

        """ Warmup """
        self.results_queue.put(f"Warm worker {self.worker_id} up for {warm_up_memory} steps")
        while self.env.current_global_step_count <= warm_up_memory:
            self.warmup_phase(max_steps=max_steps)

        self.env.current_global_step_count = 0
        self.results_queue.put(f"Warm up of {self.worker_id} finished")
        self.control_queue.put(f"WARMUP FINISHED")

        """ Send teacher current worker's velocity """
        self.control_queue.put(self.env.velocity_normalizer.maximum_value)
        """ Wait for normalizer settings """
        normalizer_settings, min_max_velocity_normalizer_settings = self.weight_queue.get()
        self.update_normalizer_settings(normalizer_settings)
        self.update_min_max_normalizer_settings(min_max_velocity_normalizer_settings)

        episode = 0
        """ Main simulation loop """
        while episode <= max_episodes:
            start_time = time.time()
            current_avg = 0.0 if len(episodes_rewards) == 0 else float(np.mean(episodes_rewards))
            try:
                msg = self.control_queue.get_nowait()
                if msg == "STOP WORKER":
                    break
            except Empty:
                pass
            reward_episode, goal_reached, steps_taken, sum_reward_components = self.run_episode(episode, init_time, current_avg)
            end_time = time.time()

            if goal_reached:
                successive_goals_reached += 1
                if self.worker_id == 0:
                    self.training_logs.append("Target reached")
            else:
                successive_goals_reached = 0

            """ Glitch logic handling """
            if steps_taken == 0:
                continue

            episodes_rewards.append(reward_episode)
            avg_step_reward = round(reward_episode / max(steps_taken, 1), 2)
            elapsed_time = round(end_time - start_time, 2)
            noise_val = self.noise.std_dev[0]
            avg_reward = round(np.mean(episodes_rewards), 5)
            log_msg = (
                f"W: {self.worker_id}; R: {[round(x, 2) for x in sum_reward_components]}; "
                f"Ep * {episode} * - R: {reward_episode}; Steps {steps_taken}; "
                f"Avg current ep => {avg_step_reward}; Avg per 1000 => {avg_reward}; "
                f"Noise: {noise_val}; Time ==> {elapsed_time}"
            )

            self.results_queue.put_nowait(log_msg)
            if self.worker_id == 0:
                self.training_logs.append(log_msg)
            avg_reward_small = round(np.mean(list(episodes_rewards)[-200:]), 2)

            """ Tensorboard logging and model saving logic """
            if self.worker_id == 0 and steps_taken > 0:
                with self.td3.summary_writer.as_default():
                    tf.summary.scalar('Episode Reward', reward_episode, step=episode)
                    tf.summary.scalar('Avg Step Reward', reward_episode / max(steps_taken, 1), step=episode)
                if avg_reward_small > self.best_reward and episode >= 200:
                    if self.best_actor is None:
                        self.best_actor = tf.keras.models.clone_model(self.td3.actor)
                    self.best_actor.set_weights(self.td3.actor.get_weights())
                    self.best_actor.save_weights(
                        os.path.join(self.directory_name, f"hexapod_actor.h5")
                    )
                    self.best_reward = avg_reward_small
                    try:
                        self.control_queue.put_nowait("SAVE")
                    except Full:
                        self.results_queue.put_nowait(f"Control is full")
                        pass

            """ Periodic training log saving """
            if self.worker_id == 0 and episode % 100 == 0 and episode > 0 and len(self.training_logs) > 0:
                with open(os.path.join(self.directory_name, 'training_logs.txt'), "a") as f:
                    f.write("\n".join(self.training_logs) + "\n")
                self.training_logs.clear()

            """ Walking confidence measuring model's performance """
            if successive_goals_reached >= 10 and not walking_to_target:
                message = "Agent is walking to target confidently"
                if self.worker_id == 0:
                    self.training_logs.append(message)
                self.results_queue.put(message)
                walking_to_target = True

            """ Final success condition """
            if avg_reward < threshold_avg_rewards and walking_to_target:
                message = "Agent learned policy"
                if self.worker_id == 0:
                    self.training_logs.append(message)
                self.results_queue.put(message)
                break
            threshold_avg_rewards = avg_reward
            episode += 1

        if self.worker_id == 0:
            self.merge_recordings()
        self.results_queue.put(f"Worker {self.worker_id} finished")
        self.control_queue.put("STOP TEACHER")

        if self.worker_id == 0:
            config = self._get_configuration()
            with open(f"{os.path.join(self.directory_name, 'config_logs_worker.txt')}", "w") as f:
                f.write("\n".join(config))
            with open(f"{os.path.join(self.directory_name, 'training_logs.txt')}", "a") as f:
                f.write("\n".join(self.training_logs))

            if self.best_actor is None:
                self.best_actor = tf.keras.models.clone_model(self.td3.actor)
                self.best_actor.set_weights(self.td3.actor.get_weights())
                self.best_actor.save_weights(
                    os.path.join(self.directory_name, f"hexapod_actor.h5")
                )

        self.env.close()
        os._exit(0)

    @staticmethod
    def handle_text_overlay(image, start_time, episode, reward_from_episode, avg_reward, step):
        """
        Method responsible for handling text overlay which is used when scene is being captured
        :param image: screen of the scene
        :param start_time: current time
        :param episode: current episode
        :param reward_from_episode: accumulated reward from start of the episode to this moment
        :param avg_reward: average episode reward over the last 1000 episodes
        :param step: current simulation step
        :return:
        """
        now_time = datetime.now() - start_time
        hours = now_time.days * 24 + now_time.seconds // 3600
        minutes = (now_time.seconds % 3600) // 60

        text = (
            f"Ep: {episode} | {hours}h {minutes}m | "
            f"Step: {step} | "
            f"R: {round(reward_from_episode, 2)} | "
            f"Avg: {round(avg_reward, 2)}"
        )
        cv2.putText(
            image,
            text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    @staticmethod
    def start_ffmpeg_writer(path, width, height, fps=20):
        """
        Method used to run ffmpeg writer which is responsible for video creation
        :param path: current directory path
        :param width: frame width
        :param height: frame height
        :param fps: desired frames per second
        :return:
        """
        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-vcodec", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            path
        ]

        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _capture_and_write_frame(self, episode, start_time, reward_from_episode,
                                 avg_reward, current_steps, writer):
        """
        Handles vision capture, overlay, and writing to ffmpeg writer
        :return: updated writer instance
        """
        self.env.vision_sensor.handle_explicitly()
        img = self.env.vision_sensor.capture_rgb()
        img = (img * 255).astype(np.uint8)

        video_path = os.path.join(self.directory_name, f"episode_{episode}.mp4")
        if writer is None:
            h, w, _ = img.shape
            writer = self.start_ffmpeg_writer(video_path, w, h, fps=20)

        self.handle_text_overlay(
            img,
            start_time,
            episode,
            reward_from_episode,
            avg_reward,
            current_steps
        )

        writer.stdin.write(img.tobytes())
        writer.stdin.flush()

        return writer, video_path

    def run_episode(self, episode, start_time=time.time(), avg_reward=0.0):
        """
        Method responsible for running all the steps of the episode
        :param episode: current episode number
        :param start_time: episode start
        :param avg_reward: average reward over the last 1000 episodes
        :return: reward_from_episode, goal_reached, current_steps, sum_reward_components
        """
        try:
            state = self.env.reset()
            self.noise.reset_noise()

            reward_from_episode = 0
            current_steps = 0
            sum_reward_components = [0, 0, 0]
            goal_reached = False
            actor_weights = None
            writer = None
            drained = 0
            video_path = None

            """ Update the actor network from weight queue """
            try:
                while True:
                    actor_weights = self.weight_queue.get_nowait()
                    drained += 1
            except Empty:
                if actor_weights is not None:
                    self.td3.actor.set_weights(actor_weights)
                pass

            while current_steps < self.max_steps:
                self.env.current_global_step_count += 1
                actions = self._policy(state, self.noise)

                """ Set joint target positions """
                self.env.set_joint_positions(actions)

                """ Step the simulation """
                self.env.step_and_update()

                if episode % 50 == 0 and self.worker_id == 0:
                    writer, video_path = self._capture_and_write_frame(episode, start_time, reward_from_episode, avg_reward, current_steps, writer)

                current_simulation_count = 0
                """ Wait until all joints are synced"""
                while not self.env.all_joints_reached_targets():
                    current_simulation_count += 1
                    if current_simulation_count > self.env.max_simulation_count:
                        break

                    """ Step the simulation """
                    self.env.step_and_update()

                    if episode % 50 == 0 and self.worker_id == 0:
                        writer, video_path = self._capture_and_write_frame(episode, start_time, reward_from_episode, avg_reward, current_steps, writer)

                """ Handle environment glitches """
                if self.env.check_robot_collision():
                    self.results_queue.put_nowait(f"Glitch")
                    break

                next_state, reward, done, goal_reached, reward_components = self.env.get_step_results()
                sum_reward_components = [
                    s + r for s, r in zip(sum_reward_components, reward_components)
                ]

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

            reward_from_episode = round(reward_from_episode, 5)
            if episode % 50 == 0 and self.worker_id == 0:
                writer.stdin.close()
                writer.wait()
                self.all_recordings.append(video_path)

            return reward_from_episode, goal_reached, current_steps, sum_reward_components

        except Exception as e:
            self.results_queue.put_nowait(f"ERROR. Worker {self.worker_id} encountered {e}")

    def warmup_phase(self, max_steps=500):
        """
        Method reponsible for filling up the replay buffer
        :param max_steps: Maximum number of steps of each episode
        :return:
        """
        state = self.env.reset()
        current_steps = 0

        while current_steps < max_steps:
            try:
                self.env.current_global_step_count += 1
                actions = self._policy(state, self.noise, warmup_phase=True)

                """ Set joint target positions """
                self.env.set_joint_positions(actions)

                """ Step the simulation """
                self.env.step_and_update()

                current_simulation_count = 0
                """ Wait until all joints are synced"""
                while not self.env.all_joints_reached_targets():
                    current_simulation_count += 1
                    if current_simulation_count > self.env.max_simulation_count:
                        break

                    """ Step the simulation """
                    self.env.step_and_update()

                """ Handle environment glitches """
                if self.env.check_robot_collision():
                    self.results_queue.put_nowait(f"Glitch")
                    break

                next_state, reward, done, goal_reached, reward_components = self.env.get_step_results(warmup = True)

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

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _forward(self, state):
        """
        Method used for getting actions based on current state form the actor's network
        :param state: current state representation
        :return: Tensor of actions
        """
        return tf.squeeze(self.td3.actor(tf.expand_dims(state, 0)), axis=0)


    def _policy(self, state, noise, warmup_phase=False, training = True):
        """
        Method responsible returning state based on current phase, current action and added noise
        :param state: current state representation
        :param noise: instance of the OUNoise class
        :param warmup_phase: boolean flag
        :param training: boolean flag
        :return: Clipped actions
        """
        if warmup_phase:
            # Generate random joint position for each joint - in warmup phase
            legal_action = np.array([
                random.uniform(*self.env.joint_min_max[i // 6])
                for i in range(self.td3.number_of_actions)
            ], dtype=np.float32)

        else:
            normalized_state = self._normalize_observation_space(state)
            sampled_actions = self._forward(normalized_state).numpy()

            if training:
                noise = noise()

                sampled_actions = sampled_actions + noise

                self.noise.decay()

            legal_action = np.array([
                np.clip(
                    sampled_actions[i],
                    *self.env.joint_min_max[i // 6]
                )
                for i in range(self.td3.number_of_actions)
            ], dtype=np.float32)

        return legal_action

    def _normalize_observation_space(self, current_state):
        """
        Method responsible for normalizing current state representation
        :param current_state: urrent state representation
        :return: Tensor of normalized state representation
        """
        current_state = np.asarray(current_state, dtype=np.float32)

        return tf.convert_to_tensor(self.td3.running_normalizer.normalize(current_state), dtype=tf.float32)


    def merge_recordings(self):
        """
        Method used for final recording merging
        :return:
        """
        height, width = (1080, 1920)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(self.directory_name, f"merged_videos.mp4")
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

        for video_file in self.all_recordings:
            cap = cv2.VideoCapture(video_file)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break
            cap.release()
        out.release()

    def _load_normalizer_model(self):
        try:
            with open(os.path.join(self.directory_name, "running_normalizer_data.pkl"), "rb") as f:
                settings = pickle.load(f)
                self.td3.running_normalizer.adjust_settings(settings)
        except FileNotFoundError:
            print(f"File running_normalizer_data.pkl not found")
            return False

        try:
            with open(os.path.join(self.directory_name, "min_max_normalizer_data.pkl"), "rb") as f:
                self.env.velocity_normalizer.maximum_velocity_value = pickle.load(f)
        except FileNotFoundError:
            print(f"File min_max_normalizer_data.pkl not found")
            return False

        if os.path.exists(os.path.join(self.directory_name, "hexapod_actor.h5")):
            try:
                self.td3.actor_exp.load_weights(
                    os.path.join(self.directory_name, "hexapod_actor.h5")
                )
                self.td3.actor = self.td3.actor_exp
            except Exception as e:
                print(f"Weights not loaded {e}")
        else:
            print(f"File hexapod_actor.h5 not found")
            return False

        return True

    def run_test(self, max_steps=300):
        is_loaded = self._load_normalizer_model()
        if not is_loaded:
            print(f"Some of the values needed were not loaded")
            return

        run = True
        current_steps = 0

        while run:
            state = self.env.reset()

            while True:
                try:
                    actions = self._policy(state, None, warmup_phase=False, training=False)

                    self.env.set_joint_positions(actions)

                    """ Step the simulation """
                    self.env.step_and_update()

                    current_simulation_count = 0
                    """ Wait until physic simulation is synced"""
                    while not self.env.all_joints_reached_targets():
                        current_simulation_count += 1

                        if current_simulation_count > self.env.max_simulation_count:
                            break

                        """ Step the simulation """
                        self.env.step_and_update()

                    next_state, _, done, goal_reached, _ = self.env.get_step_results(warmup=False)

                    current_steps += 1

                    if current_steps >= max_steps:
                        run = False
                        break

                    if done or goal_reached:
                        break

                    state = next_state
                except PyRepError as e:
                    self.results_queue.put(f"Coppeliasim error occured {e}")
                except RuntimeError as e:
                        self.results_queue.put(f"Coppeliasim runtime error occured {e}")

        self.env.close()