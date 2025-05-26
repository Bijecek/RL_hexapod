import os
from datetime import datetime
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
        self.upper_bound = ddpg.upper_bound
        self.lower_bound = ddpg.lower_bound
        self.avg_episodes_rewards = []

        self.video_dir = 'recorded_episodes'
        self.all_recordings = []
        self.manage_directory()

        if load_weights:
            self.load_weights()

    """ Method responsible returning state based on current action and added noise"""
    def _policy(self, state, noise):

        sampled_actions = tf.squeeze(self.ddpg.actor(tf.expand_dims(state,0)))

        noise_o = noise()

        sampled_actions = sampled_actions + noise_o

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
    def _record_current_frames(self, episode, frames, start_time, reward_from_episode):
        video_path = os.path.join(self.video_dir, f'episode_{episode}.mp4')
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

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
            cv2.putText(bgr_frame, f'Episode: {episode}, training time: {hours} hours, {minutes} minutes | Episode reward: {int(reward_from_episode)}', position, font, font_scale, font_color, thickness,
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

    """ Main simulation logic """
    def run_episode(self, episode, max_steps=500, start_time=datetime.now()):
        state = self.env.reset()
        self.noise.reset_noise()

        reward_from_episode = 0
        current_steps = 0
        frames = []
        goal_reached = False

        while current_steps < max_steps:
            actions = self._policy(state, self.noise)
            next_state, reward, done, goal_reached = self.env.step(actions)

            """ Capture frame if this is the 50th episode """
            if episode % 50 == 0 and episode > 0:

                """ We use handle_explicitly to turn on the sensor for a this capture"""
                self.env.vision_sensor.handle_explicitly()
                img = self.env.vision_sensor.capture_rgb()
                img = (img * 255).astype(np.uint8)
                frames.append(img)

            self.memory.insert_to_memory((state, actions, reward, next_state, done))
            self._update_observation_space(state)
            reward_from_episode += reward

            """ Update the networks every 10 steps"""
            if self.memory.current_capacity >= self.memory.batch_size and self.memory.current_capacity % 10 == 0:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample_from_memory()
                state_batch = self._normalize_observation_space(state_batch)
                next_state_batch = self._normalize_observation_space(next_state_batch)

                state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
                action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
                reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
                next_state_batch = tf.convert_to_tensor(
                    next_state_batch, dtype=tf.float32
                )
                done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)

                self.ddpg.train(episode, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                self.ddpg.update_target_models()

            if done:
                break
            current_steps += 1
            state = next_state

        if episode % 50 == 0 and episode > 0:
            self._record_current_frames(episode, frames, start_time, reward_from_episode)

        return reward_from_episode, goal_reached
    def run_episodes(self, max_episodes=100, max_steps=500):
        episodes_rewards = []
        successive_goals_reached = 0
        init_time = datetime.now()

        for episode in range(max_episodes):
            start_time = time.time()
            reward_episode, goal_reached = self.run_episode(episode, max_steps, init_time)
            end_time = time.time()

            if goal_reached:
                successive_goals_reached += 1
            else:
                successive_goals_reached = 0

            episodes_rewards.append(reward_episode)
            avg_reward = np.mean(episodes_rewards[-50:])
            print(f"Episode * {episode} * - Reward: {reward_episode} ; Avg Reward is ==> {avg_reward} ; Ep. time is ==> {end_time - start_time}")
            self.avg_episodes_rewards.append(avg_reward)

            """ Purge cache every 500 episodes due to previous high memory consumption"""
            if episode % 500 == 0 and episode > 0:
                tf.keras.backend.clear_session()
                gc.collect()

            """ Tensorboard logging logic """
            with self.ddpg.summary_writer.as_default():
                tf.summary.scalar('Episode Reward', reward_episode, step=episode)

            if successive_goals_reached >= 5:
                print("Agent learned policy")
                break

        self.env.close()

    """ Save the model results (its weights) """
    def save_results(self):
        self.ddpg.actor.save_weights("hexapod_actor.weights.h5")
        self.ddpg.critic.save_weights("hexapod_critic.weights.h5")
        self.ddpg.target_actor.save_weights("hexapod_target_actor.weights.h5")
        self.ddpg.target_critic.save_weights("hexapod_target_critic.weights.h5")

    """ Merge separate episode recording into one video """
    def merge_recordings(self):
        height, width = (720, 1280)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter("recorded_episodes/merged_videos.mp4", fourcc, 30.0, (width, height))

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

    def load_weights(self):
        self.ddpg.actor.load_weights("Saved_weights/v2_5000ep_3500avg/hexapod_actor.weights.h5")
        self.ddpg.critic.load_weights("Saved_weights/v2_5000ep_3500avg/hexapod_critic.weights.h5")
        self.ddpg.target_actor.load_weights("Saved_weights/v2_5000ep_3500avg/hexapod_target_actor.weights.h5")
        self.ddpg.target_critic.load_weights("Saved_weights/v2_5000ep_3500avg/hexapod_target_critic.weights.h5")



