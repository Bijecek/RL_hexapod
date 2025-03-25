import numpy as np

from DDPG import DDPG
from DDPG.Memory import Memory
import tensorflow as tf
import matplotlib.pyplot as plt


from DDPG.Noise import Noise


#tf.compat.v1.enable_eager_execution()


class Agent:
    def __init__(self, env, ddpg: DDPG, memory : Memory, noise: Noise):
        self.env = env
        self.ddpg = ddpg
        self.memory = memory
        self.noise = noise
        self.upper_bound = ddpg.upper_bound
        self.lower_bound = ddpg.lower_bound
        self.avg_episodes_rewards = []

    # Choose the next action that the agent should take
    def policy(self, state, noise):

        # TODO NORMALIZACE
        #normalized_state = np.array(self.ddpg.normalizer.normalize(state)).reshape(1, -1)

        # TODO NORMALIZACE
        #state = state / self.upper_bound
        state_positions = np.array(self.ddpg.normalizer_positions.normalize(state[:18]))
        state_velocities = np.array(self.ddpg.normalizer_velocities.normalize(state[18:]))

        #print(state_positions.shape)
        state = np.concatenate([state_positions, state_velocities])
        #print(state.shape)

        sampled_actions = tf.squeeze(self.ddpg.actor(tf.expand_dims(state,0)))
        #sampled_actions = self.ddpg.actor(state)
        #print(sampled_actions.shape)
        noise_o = noise()
        #print("Noise",noise_o)
        #print("Sampl:",sampled_actions)
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise_o
        #print(sampled_actions)

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.ddpg.lower_bound, self.ddpg.upper_bound)
        #print(legal_action.shape)

        return legal_action


        # normalized_state = self.ddpg.normalizer.normalize(state)
        #
        # state_tensor = tf.expand_dims(tf.convert_to_tensor(normalized_state), 0)
        # noise_obj = noise()
        #
        # sampled_actions = tf.squeeze(self.ddpg.actor(state_tensor))
        #
        #
        # sampled_actions = sampled_actions.numpy() + noise_obj
        #
        # chosen_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        # return chosen_action

    def run_episode(self, max_steps=500):
        # TODO: UNCOMMENT
        #state, info = self.env.reset()

        # TODO UNCOMMENT
        state = self.env.reset()
        #state, _ = self.env.reset()

        reward_from_episode = 0

        current_steps = 0

        #print(state)

        while current_steps < max_steps:
            actions = self.policy(state, self.noise)
            #if current_steps == 0:
                #print("First:", actions)

            next_state, reward, done, _ = self.env.step(actions)
            # TODO: UNCOMMENT
            #next_state, reward, done, truncated, _ = self.env.step(actions)


            self.memory.insert_to_memory((state, actions, reward, next_state))
            reward_from_episode += reward

            # TODO: TRAIN every 10 steps
            #if current_steps > 0 and current_steps % 10 == 0:
            if self.memory.current_capacity >= self.memory.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample_from_memory()
                self.ddpg.train(state_batch, action_batch, reward_batch, next_state_batch)
                self.ddpg.update_target_models()

                # Not sure
                #self.memory.current_capacity = 0
            if done:
                break
            current_steps += 1
            state = next_state
        return reward_from_episode
    def run_episodes(self, total_episodes=100):
        episodes_rewards = []

        for episode in range(total_episodes):
            reward_episode = self.run_episode()
            episodes_rewards.append(reward_episode)
            avg_reward = np.mean(episodes_rewards[-20:])
            print(f"Episode * {episode} * - Reward: {reward_episode} ; Avg Reward is ==> {avg_reward}")
            self.avg_episodes_rewards.append(avg_reward)
        self.env.close()

    def save_results(self):
        # Save weights (optional)
        self.ddpg.actor.save_weights("hexapod_actor.weights.h5")
        self.ddpg.critic.save_weights("hexapod_critic.weights.h5")
        self.ddpg.target_actor.save_weights("hexapod_target_actor.weights.h5")
        self.ddpg.target_critic.save_weights("hexapod_target_critic.weights.h5")

    def print_graph(self):
        plt.plot(self.avg_episodes_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Episodic Reward")
        plt.show()



