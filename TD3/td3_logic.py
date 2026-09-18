from TD3.td3 import TD3
import tensorflow as tf
from tensorflow.keras import mixed_precision

class TD3Logic(TD3):
    def __init__(
            self,
            number_of_states,
            number_of_actions,
            action_bounds_min,
            action_bounds_max,
            actor_lr,
            critic_lr,
            gamma,
            tau,
            policy_noise=0.2,
            noise_clip=0.4,
            policy_delay=2,
            experiment_num=1
    ):
        """
        Initialization of the TD3Logic class
        :param number_of_states: constant representing observation space dimension
        :param number_of_actions: constant representing action space dimension
        :param action_bounds_min: list representing lower bounds of each action movement
        :param action_bounds_max: list representing upper bounds of each action movement
        :param actor_lr: constant representing the learning rate of the actor
        :param critic_lr: constant representing the learning rate of the critic
        :param gamma: constant representing the discount factor
        :param tau: constant representing the target network update rate
        :param policy_noise: constant representing the noise added to the actor's actions
        :param noise_clip: constant representing the clipping range of the actor's actions
        :param policy_delay: constant representing the delay of the network updates
        """
        super().__init__(
            number_of_states,
            number_of_actions,
            action_bounds_min,
            action_bounds_max,
            experiment_num
        )

        self.gamma = gamma
        self.tau = tau
        self.learning_rate_actor = actor_lr
        self.learning_rate_critic = critic_lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor_loss = 0.0
        self.critic_loss_1 = 0.0
        self.critic_loss_2 = 0.0
        self.avg_q_value = 0.0

        self.critic_1 = self._create_critic()
        self.critic_2 = self._create_critic()
        self.target_actor = self.actor_exp
        self.target_actor.set_weights(self.actor_exp.get_weights())

        self.target_critic_1 = self._create_critic()
        self.target_critic_2 = self._create_critic()

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.base_log_interval = 1000
        self.log_interval = None

    def _soft_update(self, source_model, target_model):
        """
        Method used to perform soft update of target network
        :param source_model: actor or critic model
        :param target_model: actor or critic target model
        :return:
        """
        for source, target in zip(source_model.variables, target_model.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)


    @tf.function(
        jit_compile=False
    )
    def _train_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, do_actor_update):
        """
        Method containing TD3 algorithm's network update logic
        :param state_batch: batch of states sampled from replay memory
        :param action_batch: batch of actions sampled from replay memory
        :param reward_batch: batch of rewards sampled from replay memory
        :param next_state_batch: batch of next states sampled from replay memory
        :param done_batch: batch of done flags sampled from replay memory
        :param do_actor_update: boolean flag that ensures policy delayed updates
        :return: critic_loss_1, critic_loss_2, actor_loss
        """
        noise = tf.clip_by_value(
            tf.random.normal(shape=tf.shape(action_batch), stddev=self.policy_noise),
            -self.noise_clip,
            self.noise_clip
        )

        next_action = self.target_actor(next_state_batch)
        next_action = tf.clip_by_value(
            next_action + noise,
            self.action_bounds_min,
            self.action_bounds_max
        )

        target_q1 = self.target_critic_1([next_state_batch, next_action])
        target_q2 = self.target_critic_2([next_state_batch, next_action])
        target_q = tf.minimum(target_q1, target_q2)

        """ Added done_batch flag to differentiate between state that has no further state """
        y = tf.stop_gradient(
            reward_batch + (1.0 - tf.cast(done_batch, tf.float32)) * self.gamma * target_q
        )

        """ First critic training logic """
        with tf.GradientTape() as tape1:
            q1 = self.critic_1([state_batch, action_batch])
            critic_loss_1 = tf.reduce_mean(tf.square(y - q1))
        grads1 = tape1.gradient(critic_loss_1, self.critic_1.trainable_variables)

        """ Second critic training logic """
        with tf.GradientTape() as tape2:
            q2 = self.critic_2([state_batch, action_batch])
            critic_loss_2 = tf.reduce_mean(tf.square(y - q2))
        grads2 = tape2.gradient(critic_loss_2, self.critic_2.trainable_variables)

        """ Gradient clipping """
        grads1, _ = tf.clip_by_global_norm(grads1, 1.0)
        grads2, _ = tf.clip_by_global_norm(grads2, 1.0)

        self.critic_1_optimizer.apply_gradients(zip(grads1, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(grads2, self.critic_2.trainable_variables))

        actor_loss = 0.0
        """ Delayed network's update """
        if do_actor_update:
            with tf.GradientTape() as tape:
                actions = self.actor(state_batch)
                actor_loss = -tf.reduce_mean(self.critic_1([state_batch, actions]))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, 1.0)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            """ Soft updates """
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic_1, self.target_critic_1)
            self._soft_update(self.critic_2, self.target_critic_2)

        return critic_loss_1, critic_loss_2, actor_loss

    def train(self, num_of_network_updates, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """
        Method responsible for TensorFlow logging and network delayed policy update calling
        :param num_of_network_updates: counter representing the number of all network updates
        :param state_batch: batch of states sampled from replay memory
        :param action_batch: batch of actions sampled from replay memory
        :param reward_batch: batch of rewards sampled from replay memory
        :param next_state_batch: batch of next states sampled from replay memory
        :param done_batch: batch of done flags sampled from replay memory
        :return:
        """
        """ Not convex shaping ensures that the first call will initialize all tf.Variables """
        do_actor_update = (num_of_network_updates % self.policy_delay == 1)

        critic_loss_1, critic_loss_2, actor_loss = self._train_step(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, do_actor_update
        )

        self.critic_loss_1 = float(critic_loss_1)
        self.critic_loss_2 = float(critic_loss_2)
        if do_actor_update:
            self.actor_loss = float(actor_loss)
            self.avg_q_value = float(-actor_loss)

        if num_of_network_updates % self.log_interval == 0:
            """ Tensorboard logging"""
            with self.summary_writer.as_default():
                tf.summary.scalar('Actor Loss', self.actor_loss, step=num_of_network_updates)
                tf.summary.scalar('Critic Loss 1', self.critic_loss_1, step=num_of_network_updates)
                tf.summary.scalar('Critic Loss 2', self.critic_loss_2, step=num_of_network_updates)
                tf.summary.scalar('Q-Value', self.avg_q_value, step=num_of_network_updates)