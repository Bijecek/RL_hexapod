import numpy as np

""" Class responsible for generating Ornstein-Uhlenbeck noise"""
class Noise:
    def __init__(self, mean, std_deviation, theta, dt, decay_constant):
        self.x_prev = None
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.reset_noise()
        self.decay_constant = decay_constant

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        """ Make the next noise depend on the current one"""
        self.x_prev = x
        return x

    def reset_noise(self):
        self.x_prev = np.zeros_like(self.mean)

    def decay(self, dimension):
        self.std_dev = round(float(self.std_dev[0] * self.decay_constant), 5) * np.ones(dimension)
