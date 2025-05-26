import numpy as np

""" Class ensuring running normalization of data that is being sampled out of memory """
class RunningNormalizer:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4
        self.epsilon = epsilon
        self.shape = shape

    """ Update logic implementing Welford's algorithm which provides stable mean and variance updates"""
    def update(self, batch_data):
        batch_mean = np.mean(batch_data, axis=0)
        batch_var = np.var(batch_data, axis=0)
        batch_count = batch_data.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        previous = self.var * self.count
        current = batch_var * batch_count
        correction = (delta ** 2) * self.count * batch_count / total_count
        new_var = (previous + current + correction) / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    """ Method handling input data normalization"""
    def normalize(self, state):
        return (state - self.mean) / np.sqrt(self.var + self.epsilon)


