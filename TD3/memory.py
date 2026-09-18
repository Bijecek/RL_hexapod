import numpy as np

class Memory:
    def __init__(self, max_capacity, number_of_states, number_of_actions, batch_size):
        """
        Circular buffer memory implementation
        :param max_capacity: Constant of max memory capacity
        :param number_of_states: constant representing observation space dimension
        :param number_of_actions: constant representing action space dimension
        :param batch_size: constant of chosen batch size samples
        """
        self.max_capacity = max_capacity
        self.current_capacity = 0
        self.current_insert_position = 0
        self.batch_size = batch_size
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

        self.state_memory = np.zeros((self.max_capacity, number_of_states), dtype=np.float32)
        self.action_memory = np.zeros((self.max_capacity, number_of_actions), dtype=np.float32)
        self.reward_memory = np.zeros((self.max_capacity,1), dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_capacity, number_of_states), dtype=np.float32)
        self.done_memory = np.zeros((self.max_capacity, 1), dtype=np.bool_)

    def insert_to_memory(self, observation):
        """
        Method responsible for insert a sample into memory
        :param observation: current observation
        :return:
        """
        self.state_memory[self.current_insert_position] = observation[0]
        self.action_memory[self.current_insert_position] = observation[1]
        self.reward_memory[self.current_insert_position] = observation[2]
        self.next_state_memory[self.current_insert_position] = observation[3]
        self.done_memory[self.current_insert_position] = observation[4]


        self.current_capacity = min(self.current_capacity + 1, self.max_capacity)

        self.current_insert_position = (self.current_insert_position + 1) % self.max_capacity

    def sample_from_memory(self):
        """
        Method responsible for sampling batch_size of samples from memory
        :return: batch of observations containing state_batch, action_batch, reward_batch, next_state_batch, done_batch
        """
        record_range = min(self.current_capacity, self.max_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size, replace=True)

        state_batch = self.state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        done_batch = self.done_memory[batch_indices]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch