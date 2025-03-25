import numpy as np
import tensorflow as tf
import keras


class Memory:
    def __init__(self, max_capacity, number_of_states, number_of_actions):
        self.max_capacity = max_capacity
        self.current_capacity = 0
        self.batch_size = 1024

        self.state_memory = np.zeros((self.max_capacity, number_of_states))
        self.action_memory = np.zeros((self.max_capacity, number_of_actions))
        self.reward_memory = np.zeros((self.max_capacity,1))
        #self.reward_memory = np.zeros(self.max_capacity)
        self.next_state_memory = np.zeros((self.max_capacity, number_of_states))

    def insert_to_memory(self, observation):
        # Circular memory logic
        current_index = self.current_capacity%self.max_capacity

        self.state_memory[current_index] = observation[0]
        self.action_memory[current_index] = observation[1]
        self.reward_memory[current_index] = observation[2]
        self.next_state_memory[current_index] = observation[3]

        self.current_capacity += 1

    # We compute the loss and update parameters
    def sample_from_memory(self):
        # Get sampling range
        record_range = min(self.current_capacity, self.max_capacity)

        #batch_size_tmp = min(self.batch_size, record_range)
        #batch_size_tmp = self.batch_size
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_memory[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_memory[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_memory[batch_indices], dtype="float32")
        next_state_batch = tf.convert_to_tensor(
            self.next_state_memory[batch_indices]
        )

        return state_batch, action_batch, reward_batch, next_state_batch

