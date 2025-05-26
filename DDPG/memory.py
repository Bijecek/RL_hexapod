import numpy as np

""" Circular buffer memory implementation """
class Memory:
    def __init__(self, max_capacity, number_of_states, number_of_actions):
        self.max_capacity = max_capacity
        self.current_capacity = 0
        self.batch_size = 1028

        self.state_memory = np.zeros((self.max_capacity, number_of_states), dtype=np.float32)
        self.action_memory = np.zeros((self.max_capacity, number_of_actions), dtype=np.float32)
        self.reward_memory = np.zeros((self.max_capacity,1), dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_capacity, number_of_states), dtype=np.float32)
        self.done_memory = np.zeros((self.max_capacity, 1), dtype=np.int32)

    """ Insert a sample into memory """
    def insert_to_memory(self, observation):
        current_index = self.current_capacity%self.max_capacity

        self.state_memory[current_index] = observation[0]
        self.action_memory[current_index] = observation[1]
        self.reward_memory[current_index] = observation[2]
        self.next_state_memory[current_index] = observation[3]
        self.done_memory[current_index] = observation[4]

        self.current_capacity += 1

    """ Sample batch_size of samples from memory """
    def sample_from_memory(self):
        record_range = min(self.current_capacity, self.max_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = self.state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        done_batch = self.done_memory[batch_indices]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

