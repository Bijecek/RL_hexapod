import numpy as np

class RunningNormalizer:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.M2 = np.zeros(shape, dtype=np.float32)
        self.count = 0
        self.epsilon = epsilon

    def update(self, x):
        """Update mean/variance for a single observation x (Welford algorithm)."""
        x = np.asarray(x, dtype=np.float32)
        self.count += 1

        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        if self.count < 2:
            return np.ones_like(self.mean)
        return self.M2 / (self.count - 1)

    """ Method handling input data normalization """
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.variance()) + self.epsilon)
