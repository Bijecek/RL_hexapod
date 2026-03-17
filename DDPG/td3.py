from DDPG.running_normalizer import RunningNormalizer
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, BatchNormalization, LayerNormalization, Activation

class TD3:
    def __init__(
        self,
        number_of_states,
        number_of_actions,
        upper_bound,
        lower_bound
    ):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.actor = self.create_actor()

        self.critic_1 = self.create_critic()
        self.critic_2 = self.create_critic()

        self.running_normalizer = RunningNormalizer((number_of_states[0],))

        log_dir = "logs/ddpg"
        self.summary_writer = tf.summary.create_file_writer(log_dir)


    """ Method containing actor network architecture"""

    def create_actor(self):
        he = tf.keras.initializers.HeNormal()
        last_init = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=123)

        actor_model = tf.keras.Sequential(
            [
                Input(shape=self.number_of_states),
                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                LayerNormalization(),
                Activation("relu"),
                # concat = LayerNormalization()(concat)
                # BatchNormalization(),
                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                Activation("relu"),

                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                Activation("relu"),


                Dense(self.number_of_actions[0], activation="tanh", kernel_initializer=last_init)
            ]
        )
        """ Scaling the output due to difference between tanh (-1 ; 1) range and action range """
        actor_model.add(Lambda(lambda x: x * self.upper_bound))

        return actor_model

    """ Method containing critic network architecture"""

    def create_critic(self):
        he = tf.keras.initializers.HeNormal()
        last_init = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=123)

        state_input = Input(shape=self.number_of_states)
        state_out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(state_input)
        # state_out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(state_out)
        # state_out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(state_out)

        # state_out = BatchNormalization()(state_out)

        action_input = Input(shape=self.number_of_actions)
        action_out = Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(action_input)
        # action_out = BatchNormalization()(action_out)

        concat = Concatenate()([state_out, action_out])

        out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(concat)
        out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(out)
        # out = Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(out)
        #out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(out)

        outputs = Dense(1, kernel_initializer=last_init)(out)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model
