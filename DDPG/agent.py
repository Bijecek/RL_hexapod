import copy
import io
import os
import pickle
from collections import deque, defaultdict
from contextlib import redirect_stdout
from datetime import datetime
import random

#from DDPG.ddpg import DDPG
from DDPG.td3 import TD3
from DDPG.memory import Memory
import tensorflow as tf
import gc
from DDPG.ou_noise import Noise
import time
import numpy as np
import cv2

""" Class containing main simulation logic """
class Agent:
    def __init__(self, env, td3: TD3, memory : Memory, noise: Noise, load_weights=False, directory_name=None):
        self.env = env
        self.td3 = td3
        self.memory = memory
        self.noise = noise
        #self.avg_episodes_rewards = []

        if load_weights:
            self.load_weights()

        self.best_reward = float('-inf')
        self.best_actor = None
        self.best_critic_1 = None
        self.best_critic_2 = None

        self.training_logs = []
        self.directory_name = directory_name
        self.config_logs = []

        self.max_episodes = 0
        self.max_steps = 0
        self.warm_up_memory = 0

        self.video_dir = 'recorded_episodes'
        self.all_recordings = []
        self.manage_directory()


        self.global_total_steps = 0

    def _get_configuration(self):
        config_dict = defaultdict(int)
        config_dict["Learning rate actor"] = self.td3.learning_rate_actor
        config_dict["Learning rate critic"] = self.td3.learning_rate_critic
        config_dict["Gamma"] = self.td3.gamma
        config_dict["Tau"] = self.td3.tau
        config_dict["Memory capacity"] = self.memory.max_capacity
        config_dict["Batch size"] = self.memory.batch_size
        config_dict["Theta"] = self.noise.theta
        config_dict["Dt"] = self.noise.dt
        config_dict["Decay constant"] = self.noise.decay_constant
        config_dict["Max episodes"] = self.max_episodes
        config_dict["Max steps"] = self.max_steps
        config_dict["Warm up steps"] = self.warm_up_memory

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

    @tf.function
    def _forward(self, state):
        return tf.squeeze(self.td3.actor(tf.expand_dims(state, 0)), axis=0)

    """ Method responsible returning state based on current action and added noise"""
    def _policy(self, state, noise, episode, warmup_phase=False, training = True):
        if warmup_phase:
            # Generate random joint position for each joint - in warmup phase
            legal_action = np.array([
                random.uniform(self.td3.lower_bound, self.td3.upper_bound)
                #random.uniform(0.0, 0.0)
                for _ in range(self.td3.number_of_actions[0])
            ], dtype=np.float32)

        else:
            normalized_state = self._normalize_observation_space(state)

            sampled_actions = self._forward(normalized_state)

            if training:
                noise_o = noise()

                sampled_actions = sampled_actions.numpy() + noise_o

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

    """ Video recording logic, it is used to make a video out of frames which are coming from the vision sensor in CoppeliaSim"""
    def _record_current_frames(self, episode, frames, start_time, reward_from_episode, avg_reward):
        os.makedirs(self.directory_name, exist_ok=True)

        video_path = os.path.join(self.directory_name, f"episode_{episode}.avi")
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

    """ Helper method, used to clear the current recording folder"""
    def manage_directory(self):
        os.makedirs(self.directory_name, exist_ok=True)

        for filename in os.listdir(self.directory_name):
            file_path = os.path.join(self.directory_name, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def warmup_phase(self, episode, max_steps=500):
        state = self.env.reset()
        self.noise.reset_noise()

        current_steps = 0

        episode_states = []
        while current_steps < max_steps:
            #self._update_observation_space(state)
            actions = self._policy(state, self.noise, episode, warmup_phase=True)

            """ Joint target positions set """
            self.env.set_joint_positions(actions, episode)

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

            next_state, reward, done, goal_reached = self.env.get_step_results(episode, warmup = True)
            episode_states.append(state)
            self.memory.insert_to_memory((state, actions, reward, next_state, done))

            if done or goal_reached:
                break
            current_steps += 1
            state = next_state

        self._update_observation_space(episode_states)

    """ Main simulation logic """
    def run_episode(self, episode, start_time=datetime.now(), avg_reward=0.0):
        state = self.env.reset()
        self.noise.reset_noise()

        reward_from_episode = 0
        current_steps = 0
        frames = []
        goal_reached = False

        episode_states = []
        #print("-----------------")
        while current_steps < self.max_steps:
            self.env.current_global_step_count += 1
            #self._update_observation_space(state)
            actions = self._policy(state, self.noise, episode)

            """ Joint target positions set """
            self.env.set_joint_positions(actions, episode)

            """ Step the simulation """
            self.env.step_and_update()

            """ Capture frame if this is the 50th episode """
            if episode % 200 == 0:
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

                """ Capture frame if this is the 50th episode """
                if episode % 200 == 0:
                    """ We use handle_explicitly to turn on the sensor for a this capture"""
                    self.env.vision_sensor.handle_explicitly()
                    img = self.env.vision_sensor.capture_rgb()
                    img = (img * 255).astype(np.uint8)
                    frames.append(img)

            next_state, reward, done, goal_reached = self.env.get_step_results(episode)

            episode_states.append(state)

            self.memory.insert_to_memory((state, actions, reward, next_state, done))

            reward_from_episode += reward

            self.update_network(episode)


            current_steps += 1

            if done or goal_reached:
                break

            state = next_state

        """ Update running normalizer after episode ends"""
        self._update_observation_space(episode_states)

        # print(self.ddpg.running_normalizer.mean)
        # print(episode_states[0])
        # print(self.ddpg.running_normalizer.normalize(episode_states[0]))
        # print("---------")

        reward_from_episode = round(reward_from_episode, 5)
        if episode % 200 == 0:
            self._record_current_frames(episode, frames, start_time, reward_from_episode, avg_reward)

        return reward_from_episode, goal_reached, current_steps

    def run_test_episode(self, max_steps=500):
        state = self.env.reset()

        reward_from_episode = 0
        current_steps = 0
        goal_reached = False

        while current_steps < max_steps:
            actions = self._policy(state, self.noise, 0, training=False)
            next_state, reward, done, goal_reached = self.env.step(actions, None)

            reward_from_episode += reward

            if done:
                break
            current_steps += 1
            state = next_state

        reward_from_episode = round(reward_from_episode, 5)

        return reward_from_episode, goal_reached


    def update_network(self, episode, termination_update=False):
        """ Update the networks every X steps"""
        #if self.memory.current_capacity >= self.memory.batch_size and self.memory.current_capacity % 5 == 0:
        # if self.network_iteration == 1 and self.memory.current_capacity > 100_000:
        #     self.network_iteration = 4
        #     print(f"Network iteration changed to {self.network_iteration}")

        if (self.memory.current_capacity >= self.memory.batch_size and self.memory.current_capacity % 2 == 0) or termination_update:
            self.global_total_steps += 1
            #print(self.memory.current_capacity)
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


    def run_episodes(self, max_episodes=100, max_steps=500, warm_up_memory=500, preload_warmup=False):
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.warm_up_memory = warm_up_memory

        episodes_rewards = deque(maxlen=1000)
        successive_goals_reached = 0
        init_time = datetime.now()

        if preload_warmup:
            print(f"Warm up - loading serialized memory")
            with open(f"{os.path.join('warmup_data_1209_dynamic', 'memory_data.pkl')}", "rb") as f:
                new_memory = pickle.load(f)
                self.memory.state_memory = copy.deepcopy(new_memory.state_memory)
                self.memory.action_memory = copy.deepcopy(new_memory.action_memory)
                self.memory.reward_memory = copy.deepcopy(new_memory.reward_memory)
                self.memory.next_state_memory = copy.deepcopy(new_memory.next_state_memory)
                self.memory.done_memory = copy.deepcopy(new_memory.done_memory)
                self.memory.current_capacity = copy.deepcopy(new_memory.current_capacity)


            print(f"Warm up finished - memory loaded")

            with open(f"{os.path.join('warmup_data_1209_dynamic', 'running_normalizer_data.pkl')}", "rb") as f:
                self.td3.running_normalizer = pickle.load(f)
                self.env.distance_normalizer = pickle.load(f)
                self.env.velocity_normalizer = pickle.load(f)
            #     self.env.body_position_normalizer = pickle.load(f)

            print("Warm up finished - normalizers loaded")
        else:
            print(f"Warm up for {warm_up_memory} steps")
            while self.memory.current_capacity <= warm_up_memory:
                self.warmup_phase(None, max_steps=max_steps)

            with open("memory_data.pkl", "wb") as f:
                pickle.dump(self.memory, f)

            with open("running_normalizer_data.pkl", "wb") as f:
                pickle.dump(self.td3.running_normalizer, f)
                pickle.dump(self.env.distance_normalizer, f)
                pickle.dump(self.env.velocity_normalizer, f)
                # pickle.dump(self.env.body_position_normalizer, f)

            print(f"Warm up finished in { ((datetime.now() - init_time).total_seconds()) / 60} minutes")

        threshold_avg_rewards = [float('-inf')]
        #print(f"Min: {self.env.reward_min} | Max: {self.env.reward_max}")
        for episode in range(max_episodes+1):
            if self.env.current_global_step_count >= self.env.phase_limit:
                if self.env.current_phase < self.env.max_phase_num:
                    self.env.current_phase += 1
                    print("Moving to the next phase, current phase is: ", self.env.current_phase)
                    print("Clearing memory")
                    self.memory.clear_memory()
                    episodes_rewards.clear()
                    self.env.current_global_step_count = 0
                    print("Warmup started")
                    print(f"Warm up for {warm_up_memory} steps")
                    while self.memory.current_capacity <= warm_up_memory:
                        self.warmup_phase(None, max_steps=max_steps)
                    print("Warm up finished")

                    if self.env.current_phase == 3:
                        self.max_steps = 200


            start_time = time.time()
            current_avg = 0.0 if len(episodes_rewards) == 0 else float(np.mean(episodes_rewards))
            reward_episode, goal_reached, steps_taken = self.run_episode(episode, init_time, current_avg)

            end_time = time.time()

            if goal_reached:
                successive_goals_reached += 1
                self.training_logs.append("Target reached")
            else:
                successive_goals_reached = 0

            episodes_rewards.append(reward_episode)
            avg_reward = round(np.mean(episodes_rewards), 5)

            print(f"Ep * {episode} * - R: {reward_episode} ; Avg current ep => {round(reward_episode/steps_taken, 2)}; Avg per 1000 => {avg_reward} ; Phase: {self.env.current_phase} ; Time ==> {round(end_time - start_time, 2)}")
            self.training_logs.append(f"Ep * {episode} * - R: {reward_episode} ; Avg current ep => {round(reward_episode/steps_taken, 2)}; Avg per 1000 => {avg_reward} ; Phase: {self.env.current_phase} ; Time ==> {round(end_time - start_time, 2)}")

            """ Tensorboard logging logic """
            with self.td3.summary_writer.as_default():
                tf.summary.scalar('Episode Reward', reward_episode, step=episode)
                tf.summary.scalar('Avg Step Reward',reward_episode/steps_taken, step=episode)

            if successive_goals_reached >= 20:
                message = "Agent learned policy"
                self.training_logs.append(message)
                print(message)
                break
            avg_reward_small = round(np.mean(list(episodes_rewards)[-200:]), 3)
            if avg_reward_small > self.best_reward and episode >= 1000:
                self.best_reward = avg_reward_small
                self.best_actor = tf.keras.models.clone_model(self.td3.actor)
                self.best_actor.set_weights(self.td3.actor.get_weights())

                self.best_critic_1 = tf.keras.models.clone_model(self.td3.critic_1)
                self.best_critic_1.set_weights(self.td3.critic_1.get_weights())

                self.best_critic_2 = tf.keras.models.clone_model(self.td3.critic_2)
                self.best_critic_2.set_weights(self.td3.critic_2.get_weights())

                message = f"New best average reward: {self.best_reward}"
                self.training_logs.append(message)
                print(message)

            if episode % 1000 == 0 and episode > 0:
                # Noise decay
                self.noise.decay(self.env.action_space.shape)

                message = f"Noise decayed to {self.noise.std_dev[0]}"
                self.training_logs.append(message)
                print(message)

                if avg_reward < threshold_avg_rewards[-1] and episode > 15_000:
                    message = "Agent is not learning"
                    self.training_logs.append(message)
                    print(message)
                    break
                threshold_avg_rewards.append(avg_reward)

        # for mean in self.ddpg.running_normalizer.mean:
        #     print(f"Running mean: {mean}")
        self.env.close()

    """ Testing phase"""

    def run_test(self, max_episodes=10, max_steps=5000):
        for episode in range(max_episodes):
            reward_episode, goal_reached = self.run_test_episode(max_steps)
            print(f"Episode {episode} * - Reward: {reward_episode}")
        self.env.close()


    """ Save the model results (its weights) """
    def save_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        print(f"Saving models with average reward {round(self.best_reward, 5)}")
        try:
            self.best_actor.save_weights(os.path.join(directory, "hexapod_actor.h5"))
            self.best_critic_1.save_weights(os.path.join(directory, "hexapod_critic_1.h5"))
            self.best_critic_2.save_weights(os.path.join(directory, "hexapod_critic_2.h5"))
        except Exception as e:
            print("Best models were not saved, not enough training")

        print(f"Saving training logs to {os.path.join(directory, 'training_logs.txt')}")

        with open(f"{os.path.join(self.directory_name, 'training_logs.txt')}", "w") as f:
            f.write("\n".join(self.training_logs))

        print(f"Saving configurational logs to {os.path.join(directory, 'config_logs.txt')}")
        self.config_logs = self._get_configuration()
        with open(f"{os.path.join(self.directory_name, 'config_logs.txt')}", "w") as f:
            f.write("\n".join(self.config_logs))

    """ Merge separate episode recording into one video """
    def merge_recordings(self):
        #height, width = (1080, 1920)
        height, width = (720, 1280)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        video_path = os.path.join(self.directory_name, "merged_videos.avi")
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

        for video_file in self.all_recordings:
            cap = cv2.VideoCapture(video_file)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    out.write(frame)
                else:
                    break
            cap.release()

        out.release()

    def load_weights(self):
        self.td3.actor.load_weights("251114_test_4096batch_20_ep/hexapod_actor.h5")
        self.td3.critic.load_weights("251114_test_4096batch_20_ep/hexapod_critic.h5")





