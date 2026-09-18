from TD3.running_normalizer import RunningNormalizer
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, LayerNormalization, Activation

class TD3:
    def __init__(self, number_of_states, number_of_actions, action_bounds_min, action_bounds_max, testing_num):
        """
        Initialization of the TD3 class
        :param number_of_states: constant representing observation space dimension
        :param number_of_actions: constant representing action space dimension
        :param action_bounds_min: list representing lower bounds of each action movement
        :param action_bounds_max: list representing upper bounds of each action movement
        """
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.running_normalizer = RunningNormalizer((number_of_states,))
        log_dir = "logs/td3"
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.action_bounds_min = tf.convert_to_tensor(action_bounds_min, dtype=tf.float32)
        self.action_bounds_max = tf.convert_to_tensor(action_bounds_max, dtype=tf.float32)

        self.actor = self._create_actor()

        self.testing_num = testing_num
        if testing_num == 1:
            self.actor_exp = self._create_exp1_actor()
        elif testing_num == 2:
            self.actor_exp = self._create_actor()
        elif testing_num == 3:
            self.actor_exp = self._create_actor()

        self.actor = self.actor_exp

    def _create_exp1_actor(self):
        """
        Method containing creation of the actor network architecture for the first experiment only
        :return: Actor model
        """
        last_init = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=123)

        actor_model = tf.keras.Sequential(
            [
                Input(shape=(self.number_of_states,)),
                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                LayerNormalization(),
                Activation("relu"),
                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                Activation("relu"),
                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                Activation("relu"),
                Dense(self.number_of_actions, activation="tanh", kernel_initializer=last_init),

                Lambda(lambda x: self._scale_action(x))
            ]
        )
        return actor_model

    def _scale_action(self, action):
        """
        Method used for scaling each output of the actor network into correct action range
        :param action: current action before scaling
        :return: Scaled action
        """
        return self.action_bounds_min + (action + 1.0) * 0.5 * (self.action_bounds_max - self.action_bounds_min)

    def _create_actor(self):
        """
        Method containing creation of the actor network architecture
        :return: Actor model
        """
        last_init = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=123)

        actor_model = tf.keras.Sequential(
            [
                Input(shape=(self.number_of_states,)),
                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                LayerNormalization(),
                Activation("relu"),
                Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
                Activation("relu"),
                Dense(self.number_of_actions, activation="tanh", kernel_initializer=last_init),

                Lambda(lambda x: self._scale_action(x))
            ]
        )
        return actor_model

    def _create_critic(self):
        """
        Method containing creation of the critic network architecture
        :return: Critic model
        """
        last_init = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=123)

        state_input = Input(shape=(self.number_of_states,))
        state_out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(state_input)

        action_input = Input(shape=(self.number_of_actions,))
        action_out = Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(action_input)

        concat = Concatenate()([state_out, action_out])

        out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(concat)
        out = Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(out)

        outputs = Dense(1, kernel_initializer=last_init)(out)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model
