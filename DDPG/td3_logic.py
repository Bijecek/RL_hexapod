from DDPG.td3 import TD3
from DDPG.running_normalizer import RunningNormalizer
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, BatchNormalization, LayerNormalization, Activation

class TD3Logic(TD3):
    def __init__(
            self,
            number_of_states,
            number_of_actions,
            upper_bound,
            lower_bound,
            actor_lr,
            critic_lr,
            gamma,
            tau,
            policy_noise=0.2,
            noise_clip=0.4,
            policy_delay=2
    ):
        super().__init__(
            number_of_states,
            number_of_actions,
            upper_bound,
            lower_bound)

        self.target_actor = self.create_actor()
        self.target_actor.set_weights(self.actor.get_weights())

        self.target_critic_1 = self.create_critic()
        self.target_critic_2 = self.create_critic()

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.actor_loss = 0.0
        self.critic_loss_1 = 0.0
        self.critic_loss_2 = 0.0
        self.avg_q_value = 0.0

        self.gamma = gamma
        self.tau = tau

        self.learning_rate_actor = actor_lr
        self.learning_rate_critic = critic_lr

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

    def _soft_update(self, source_model, target_model):
        for source, target in zip(source_model.variables, target_model.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)


    """ Helper method containing training logic """
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(256, 53), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 18), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 53), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 1), dtype=tf.bool),
            tf.TensorSpec((), dtype=tf.bool),
        ],
        jit_compile=True,
        reduce_retracing=True
    )
    def _train_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, do_actor_update):
        noise = tf.clip_by_value(
            tf.random.normal(shape=tf.shape(action_batch), stddev=self.policy_noise),
            -self.noise_clip,
            self.noise_clip
        )

        next_action = self.target_actor(next_state_batch)
        next_action = tf.clip_by_value(
            next_action + noise,
            self.lower_bound,
            self.upper_bound
        )

        target_q1 = self.target_critic_1([next_state_batch, next_action])
        target_q2 = self.target_critic_2([next_state_batch, next_action])
        target_q = tf.minimum(target_q1, target_q2)

        """ Added done_batch flag to differentiate between state that has no further state"""
        y = reward_batch + (1.0 - tf.cast(done_batch, tf.float32)) * self.gamma * target_q

        """ Critic training logic """
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.critic_1([state_batch, action_batch])
            q2 = self.critic_2([state_batch, action_batch])
            critic_loss_1 = tf.reduce_mean((y - q1) ** 2)
            critic_loss_2 = tf.reduce_mean((y - q2) ** 2)

        grads1 = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        grads2 = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)

        grads1, _ = tf.clip_by_global_norm(grads1, 5.0)
        grads2, _ = tf.clip_by_global_norm(grads2, 5.0)

        self.critic_1_optimizer.apply_gradients(zip(grads1, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(grads2, self.critic_2.trainable_variables))

        del tape

        """ Delayed actor update"""
        if do_actor_update:
            with tf.GradientTape() as tape:
                actions = self.actor(state_batch)
                actor_loss = -tf.reduce_mean(self.critic_1([state_batch, actions]))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, 1.0)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            """ Soft update """
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic_1, self.target_critic_1)
            self._soft_update(self.critic_2, self.target_critic_2)

            return critic_loss_1, critic_loss_2, actor_loss

        return critic_loss_1, critic_loss_2, tf.constant(0.0)

    """ Method responsible for updating networks and logging """
    def train(self, global_total_steps, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """ To ensure that the first call we initialize all tf.Variables """
        do_actor_update =  (global_total_steps % self.policy_delay == 1)

        critic_loss_1, critic_loss_2, actor_loss = self._train_step(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, do_actor_update
        )

        self.critic_loss_1 = float(critic_loss_1)
        self.critic_loss_2 = float(critic_loss_2)
        if do_actor_update:
            self.actor_loss = float(actor_loss)
            self.avg_q_value = float(-actor_loss)

        """ Tensorboard logging"""
        with self.summary_writer.as_default():
            tf.summary.scalar('Actor Loss', self.actor_loss, step=global_total_steps)
            tf.summary.scalar('Critic Loss 1', self.critic_loss_1, step=global_total_steps)
            tf.summary.scalar('Critic Loss 2', self.critic_loss_2, step=global_total_steps)
            tf.summary.scalar('Q-Value', self.avg_q_value, step=global_total_steps)