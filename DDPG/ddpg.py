import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, BatchNormalization

from DDPG.running_normalizer import RunningNormalizer

""" Implementation of DDPG (Deep Deterministic Policy Gradient algorithm """
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

        """ Create networks during initialization """
        self.actor = self.create_actor()
        self.critic = self.create_critic()

        """ Create target networks during initialization """
        self.target_actor = self.create_actor()
        self.target_critic = self.create_critic()

        """ Set the weights to be identical """
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        policy = mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        """ Create optimizers """
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate_critic)
        self.critic_optimizer = mixed_precision.LossScaleOptimizer(self.critic_optimizer, dynamic=True)
        self.actor_optimizer = mixed_precision.LossScaleOptimizer(self.actor_optimizer, dynamic=True)

        self.running_normalizer = RunningNormalizer((self.number_of_states[0],))

        self.actor_loss = 0.0
        self.critic_loss = 0.0
        self.avg_q_value = 0.0

        log_dir = "logs/ddpg"
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    """ Method containing actor network architecture"""
    def create_actor(self):
        last_init = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=123)

        actor_model = tf.keras.Sequential(
            [
                Input(shape=self.number_of_states),
                Dense(200, activation="relu"),
                BatchNormalization(),
                Dense(200, activation="relu"),
                BatchNormalization(),
                Dense(200, activation="relu"),
                BatchNormalization(),
                Dense(self.number_of_actions[0], activation="tanh", kernel_initializer=last_init)
            ]
        )
        """ Scaling the output due to difference between tanh (0-1) range and action range """
        actor_model.add(Lambda(lambda x: x * self.upper_bound))

        return actor_model

    """ Method containing critic network architecture"""
    def create_critic(self):
        state_input = Input(shape=self.number_of_states)
        state_out = Dense(300, activation="relu")(state_input)
        state_out = BatchNormalization()(state_out)

        action_input = Input(shape=self.number_of_actions)
        action_out = Dense(100, activation="relu")(action_input)

        concat = Concatenate()([state_out, action_out])

        out = Dense(100, activation="relu")(concat)

        outputs = Dense(1)(out)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    """ Method handling soft updates for both target actor and target critic networks"""
    def update_target_models(self):
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

    """ Helper method containing soft update logic"""
    def _soft_update(self, source_model, target_model):
        new_weights = []
        target_variables = target_model.weights
        for i, weight in enumerate(source_model.weights):
            new_weights.append(self.tau * weight + (1 - self.tau) * target_variables[i])
        target_model.set_weights(new_weights)

    """ Helper method containing training logic """
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(1028, 135), dtype=tf.float32),
            tf.TensorSpec(shape=(1028, 18), dtype=tf.float32),
            tf.TensorSpec(shape=(1028, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(1028, 135), dtype=tf.float32),
            tf.TensorSpec(shape=(1028, 1), dtype=tf.float32)
        ],
        jit_compile=True,
        reduce_retracing=True
    )
    def _train_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """ Critic training logic """
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=False)
            target_critic_value = self.target_critic([next_state_batch, target_actions], training=False)

            """ Added done_batch flag to differentiate between state that has no further state"""
            y = reward_batch + (1.0 - done_batch) * self.gamma * target_critic_value

            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))
            scaled_critic_loss = self.critic_optimizer.get_scaled_loss(critic_loss)

        scaled_critic_grad = tape.gradient(scaled_critic_loss, self.critic.trainable_variables)
        critic_grad = self.critic_optimizer.get_unscaled_gradients(scaled_critic_grad)
        critic_grad, _ = tf.clip_by_global_norm(critic_grad, 1.0)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        """ Actor training logic """
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value_for_actor = self.critic([state_batch, actions], training=True)
            actor_loss = -tf.reduce_mean(critic_value_for_actor)
            scaled_actor_loss = self.actor_optimizer.get_scaled_loss(actor_loss)

        scaled_actor_grad = tape.gradient(scaled_actor_loss, self.actor.trainable_variables)
        actor_grad = self.actor_optimizer.get_unscaled_gradients(scaled_actor_grad)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        return critic_loss, actor_loss, tf.reduce_mean(critic_value_for_actor)

    """ Method responsible for updating networks and logging """
    def train(self, episode, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        critic_loss, actor_loss, avg_q_value = self._train_step(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )

        self.critic_loss = float(critic_loss)
        self.actor_loss = float(actor_loss)
        self.avg_q_value = float(avg_q_value)

        """ Tensorboard logging"""
        with self.summary_writer.as_default():
            tf.summary.scalar('Actor Loss', self.actor_loss, step=episode)
            tf.summary.scalar('Critic Loss', self.critic_loss, step=episode)
            tf.summary.scalar('Q-Value', self.avg_q_value, step=episode)
