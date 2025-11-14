import os
import pickle
from collections import deque
from datetime import datetime
import random

from DDPG.ddpg import DDPG
from DDPG.memory import Memory
import tensorflow as tf
import gc
from DDPG.noise import Noise
import time
import numpy as np
import cv2

""" Class containing main simulation logic """
class Agent:
    def __init__(self, env, ddpg: DDPG, memory : Memory, noise: Noise, load_weights=False):
        self.env = env
        self.ddpg = ddpg
        self.memory = memory
        self.noise = noise
        #self.avg_episodes_rewards = []

        self.video_dir = 'recorded_episodes'
        self.all_recordings = []
        self.manage_directory()

        if load_weights:
            self.load_weights()

        self.best_reward = float('-inf')
        self.best_actor = None
        self.best_critic = None

    @tf.function
    def _forward(self, state):
        return tf.squeeze(self.ddpg.actor(tf.expand_dims(state, 0)), axis=0)

    """ Method responsible returning state based on current action and added noise"""
    def _policy(self, state, noise, warmup_phase=False, training = True):
        if warmup_phase:
            # Generate random joint position for each joint - in warmup phase
            legal_action = np.array([
                random.uniform(self.ddpg.lower_bound, self.ddpg.upper_bound)
                for _ in range(self.ddpg.number_of_actions[0])
            ], dtype=np.float32)

        else:

            sampled_actions = self._forward(state)

            if training:
                noise_o = noise()

                sampled_actions = sampled_actions.numpy() + noise_o

            legal_action = np.clip(sampled_actions, self.ddpg.lower_bound, self.ddpg.upper_bound)


        return legal_action

    """ Method responsible for normalizing bathc of states"""
    def _normalize_observation_space(self, current_state):
        state = np.array(self.ddpg.running_normalizer.normalize(current_state))
        return state

    """ Method responsible for updating normalizer based on current state"""
    def _update_observation_space(self, current_state):
        self.ddpg.running_normalizer.update(current_state)

    """ Video recording logic, it is used to make a video out of frames which are coming from the vision sensor in CoppeliaSim"""
    def _record_current_frames(self, episode, frames, start_time, reward_from_episode, avg_reward):
        video_path = os.path.join(self.video_dir, f'episode_{episode}.mp4')
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            cv2.putText(bgr_frame, f'Episode: {episode}, training time: {hours} hours, {minutes} minutes | Episode reward: {int(reward_from_episode)} | Average Episode reward: {int(avg_reward)}', position, font, font_scale, font_color, thickness,
                        cv2.LINE_AA)
            out.write(bgr_frame)

        out.release()
        self.all_recordings.append(video_path)

    """ Helper method, used to clear the current recording folder"""
    def manage_directory(self):
        os.makedirs(self.video_dir, exist_ok=True)

        for filename in os.listdir(self.video_dir):
            file_path = os.path.join(self.video_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def warmup_phase(self, episode, max_steps=500):
        state = self.env.reset()
        self.noise.reset_noise()

        current_steps = 0

        while current_steps < max_steps:
            actions = self._policy(state, self.noise, warmup_phase=True)
            next_state, reward, done, goal_reached = self.env.step(actions, episode)

            self.memory.insert_to_memory((state, actions, reward, next_state, done))
            self._update_observation_space(state)

            # self.update_network(episode)

            if done:
                break
            current_steps += 1
            state = next_state

    """ Main simulation logic """
    def run_episode(self, episode, max_steps=500, start_time=datetime.now(), avg_reward=0.0):
        state = self.env.reset()
        self.noise.reset_noise()

        reward_from_episode = 0
        current_steps = 0
        frames = []
        goal_reached = False

        while current_steps < max_steps:
            actions = self._policy(state, self.noise)
            next_state, reward, done, goal_reached = self.env.step(actions, episode)

            """ Capture frame if this is the 50th episode """
            if episode % 500 == 0 and episode > 0:

                """ We use handle_explicitly to turn on the sensor for a this capture"""
                self.env.vision_sensor.handle_explicitly()
                img = self.env.vision_sensor.capture_rgb()
                img = (img * 255).astype(np.uint8)
                frames.append(img)

            self.memory.insert_to_memory((state, actions, reward, next_state, done))
            self._update_observation_space(state)
            reward_from_episode += reward

            self.update_network(episode)

            if done:
                break
            current_steps += 1
            state = next_state

        reward_from_episode = round(reward_from_episode, 5)
        if episode % 500 == 0 and episode > 0:
            self._record_current_frames(episode, frames, start_time, reward_from_episode, avg_reward)

        return reward_from_episode, goal_reached

    def run_test_episode(self, max_steps=500):
        state = self.env.reset()

        reward_from_episode = 0
        current_steps = 0
        goal_reached = False

        while current_steps < max_steps:
            actions = self._policy(state, self.noise, training=False)
            next_state, reward, done, goal_reached = self.env.step(actions, None)

            if done:
                break
            current_steps += 1
            state = next_state

        reward_from_episode = round(reward_from_episode, 5)

        return reward_from_episode, goal_reached


    def update_network(self, episode, termination_update=False):
        """ Update the networks every X steps"""
        #if self.memory.current_capacity >= self.memory.batch_size and self.memory.current_capacity % 5 == 0:
        if (self.memory.current_capacity >= self.memory.batch_size and self.memory.current_capacity % 20 == 0) or termination_update:

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


            self.ddpg.train(episode, state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    def run_episodes(self, max_episodes=100, max_steps=500, warm_up_episodes=500, preload_warmup=False):
        episodes_rewards = deque(maxlen=1000)
        successive_goals_reached = 0
        init_time = datetime.now()

        if preload_warmup:
            print(f"Warm up - loading serialized memory")
            with open("warmup_data_200/data.pkl", "rb") as f:
                new_memory = pickle.load(f)
                self.memory.state_memory = new_memory.state_memory
                self.memory.action_memory = new_memory.action_memory
                self.memory.reward_memory = new_memory.reward_memory
                self.memory.next_state_memory = new_memory.next_state_memory
                self.memory.done_memory = new_memory.done_memory
                self.memory.current_capacity = new_memory.current_capacity

            print(f"Warm up finished - memory loaded")
        else:
            print(f"Warm up for {warm_up_episodes} episodes")
            for episode in range(warm_up_episodes):
                self.warmup_phase(episode, max_steps=max_steps)

            with open("data.pkl", "wb") as f:
                pickle.dump(self.memory, f)

            print(f"Warm up finished in { ((datetime.now() - init_time).total_seconds()) / 60} minutes")

        threshold_avg_rewards = [float('-inf')]
        for episode in range(max_episodes):
            start_time = time.time()
            current_avg = 0.0 if len(episodes_rewards) == 0 else float(np.mean(episodes_rewards))
            reward_episode, goal_reached = self.run_episode(episode, max_steps, init_time, current_avg)

            end_time = time.time()

            if goal_reached:
                successive_goals_reached += 1
            else:
                successive_goals_reached = 0

            episodes_rewards.append(reward_episode)
            avg_reward = round(np.mean(episodes_rewards), 5)

            print(f"Episode * {episode} * - Reward: {reward_episode} ; Avg Reward is ==> {avg_reward} ; Ep. time is ==> {round(end_time - start_time, 2)}")
            #self.avg_episodes_rewards.append(avg_reward)

            # """ Purge cache every 500 episodes due to previous high memory consumption"""
            # if episode % 500 == 0 and episode > 0:
            #     #tf.keras.backend.clear_session()
            #     gc.collect()

            """ Tensorboard logging logic """
            # with self.ddpg.summary_writer.as_default():
            #     tf.summary.scalar('Episode Reward', reward_episode, step=episode)

            # if successive_goals_reached >= 20:
            #     print("Agent learned policy")
            #     break
            avg_reward_small = round(np.mean(list(episodes_rewards)[-200:]), 3)
            if avg_reward_small > self.best_reward and episode >= 1000:
                self.best_reward = avg_reward_small
                self.best_actor = tf.keras.models.clone_model(self.ddpg.actor)
                self.best_actor.set_weights(self.ddpg.actor.get_weights())

                self.best_critic = tf.keras.models.clone_model(self.ddpg.critic)
                self.best_critic.set_weights(self.ddpg.critic.get_weights())

                print(f"New best average reward: {self.best_reward}")

            if episode % 1000 == 0 and episode > 0:
                # Noise decay
                self.noise.decay(self.env.action_space.shape)
                print(f"Noice decayed to {self.noise.std_dev[0]}")

                if avg_reward < threshold_avg_rewards[-1] and episode > 3000:
                    print("Agent is not learning")
                    break
                threshold_avg_rewards.append(avg_reward)

        self.env.close()

    """ Testing phase"""

    def run_test(self, max_episodes=10, max_steps=5000):
        for episode in range(max_episodes):
            reward_episode, goal_reached = self.run_test_episode(max_steps)

        self.env.close()


    """ Save the model results (its weights) """
    def save_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        # self.ddpg.actor.save_weights(f"{directory}/hexapod_actor.weights.h5")
        # self.ddpg.critic.save_weights(f"{directory}/hexapod_critic.weights.h5")
        # self.ddpg.target_actor.save_weights(f"{directory}/hexapod_target_actor.weights.h5")
        # self.ddpg.target_critic.save_weights(f"{directory}/hexapod_target_critic.weights.h5")

        print(f"Saving models with average reward {round(self.best_reward, 5)}")
        self.best_actor.save_weights(f"{directory}/hexapod_actor.h5")
        self.best_critic.save_weights(f"{directory}/hexapod_critic.h5")

    """ Merge separate episode recording into one video """
    def merge_recordings(self):
        #height, width = (1080, 1920)
        height, width = (720, 1280)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter("recorded_episodes/merged_videos.mp4", fourcc, 20.0, (width, height))

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
        self.ddpg.actor.load_weights("251112_test_4096batch_30_ep/hexapod_actor.h5")
        self.ddpg.critic.load_weights("251112_test_4096batch_30_ep/hexapod_critic.h5")

        #
        # self.ddpg.actor.load_weights(" ")
        # self.ddpg.critic.load_weights("Saved_weights/v2_5000ep_3500avg/hexapod_critic.weights.h5")
        # self.ddpg.target_actor.load_weights("Saved_weights/v2_5000ep_3500avg/hexapod_target_actor.weights.h5")
        # self.ddpg.target_critic.load_weights("Saved_weights/v2_5000ep_3500avg/hexapod_target_critic.weights.h5")



