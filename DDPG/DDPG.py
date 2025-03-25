import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Lambda
from keras.src.layers import Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler

from DDPG.RunningNormalizer import RunningNormalizer


class DDPG:
    def __init__(self, number_of_states, number_of_actions, upper_bound, lower_bound,
                 learning_rate_actor, learning_rate_critic, gamma, tau):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.tau = tau

        # Create networks
        self.actor = self.create_actor()
        self.critic = self.create_critic()
        #The target networks are copies of the original actor and critic networks, but they are
        # updated more slowly using a technique called soft updates (via the tau parameter).
        # This prevents large changes in the target value and allows for more stable
        # learning
        self.target_actor = self.create_actor()
        self.target_critic = self.create_critic()

        # Initialize target weights to match main networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Define optimizers
        self.actor_optimizer = keras.optimizers.Adamax(learning_rate_actor)
        self.critic_optimizer = keras.optimizers.Adamax(learning_rate_critic)

        # self.normalizer_positions = RunningNormalizer(number_of_states)
        # self.normalizer_velocities = RunningNormalizer(number_of_states)

        self.normalizer_positions = RunningNormalizer((18,))
        self.normalizer_velocities = RunningNormalizer((18,))

    def create_actor(self):
        last_init = keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=123)
        #last_init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=123)

        actor_model = keras.Sequential(
            [
                Input(shape=self.number_of_states),
                Dense(512, activation="relu"),
                #BatchNormalization(),
                Dense(512, activation="relu"),
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(self.number_of_actions[0], activation="tanh", kernel_initializer=last_init)
                #Dense(self.number_of_actions[0], activation="tanh")
            ]
        )

        # Add scaling of the output
        actor_model.add(Lambda(lambda x: x * self.upper_bound))

        return actor_model

    def create_critic(self):
        # state_input = Input(shape=self.number_of_states)
        # action_input = Input(shape=self.number_of_actions)
        #
        # state_model = keras.Sequential([
        #     state_input,
        #     Dense(50, activation="relu"),
        #     Dense(50, activation="relu"),
        # ])
        # action_model = keras.Sequential([
        #     action_input,
        #     Dense(50, activation="relu"),
        #     Dense(50, activation="relu"),
        # ])
        #
        # # Concentrate both models
        # # Concatenate state and action model outputs
        # concatenated = Concatenate(axis=1)([state_model.output, action_model.output])
        #
        # # Add the remaining layers
        # x = Dense(256, activation="relu")(concatenated)
        # x = Dense(256, activation="relu")(x)
        # critic_output = Dense(1, activation="linear")(x)
        #
        # # Create the critic model with inputs and outputs
        # critic_model = Model(inputs=[state_input, action_input], outputs=critic_output)
        #
        # return critic_model

        state_input = Input(shape=self.number_of_states)
        state_out = Dense(256, activation="relu")(state_input)
        state_out = Dense(256, activation="relu")(state_out)
        # Action as input
        action_input = Input(shape=self.number_of_actions)
        action_out = Dense(256, activation="relu")(action_input)
        action_out = Dense(256, activation="relu")(action_out)

        # Both are passed through separate layer before concatenating
        concat = Concatenate()([state_out, action_out])

        out = Dense(256, activation="relu")(concat)
        out = Dense(256, activation="relu")(out)
        out = Dense(256, activation="relu")(out)
        outputs = Dense(1)(out)

        # Outputs single value for give state-action
        model = keras.Model([state_input, action_input], outputs)

        return model

    def update_target_models(self):
        # Soft update target networks
        new_weights = []
        target_variables = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            new_weights.append(self.tau * weight + (1 - self.tau) * target_variables[i])
        self.target_actor.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            new_weights.append(self.tau * weight + (1 - self.tau) * target_variables[i])
        self.target_critic.set_weights(new_weights)

    @tf.function
    def train(self, state_batch, action_batch, reward_batch, next_state_batch):
        # TODO: Pridana normalizace
        #
        #self.normalizer.update(state_batch)  # <-- Add this line

        # Normalize states and next states
        #norm_states = self.normalizer.normalize(state_batch)  # <-- Normalize
        #norm_next_states = self.normalizer.normalize(next_state_batch)  # <-- Normalize

        norm_states = state_batch / self.upper_bound
        norm_next_states = next_state_batch / self.upper_bound
        norm_actions = action_batch / self.upper_bound

        # Train critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(norm_next_states, training=True)
            y = reward_batch + self.gamma * self.target_critic([norm_next_states, target_actions], training = True)
            critic_value = self.critic([norm_states, norm_actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grad = [tf.clip_by_value(grad, -1.0, 1.0) for grad in critic_grad]
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Train Actor
        with tf.GradientTape() as tape:
            actions = self.actor(norm_states, training=True)
            critic_value = self.critic([norm_states, actions], training=True)
            # We want to maximize the expected value so minimize -value
            actor_loss = -tf.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))