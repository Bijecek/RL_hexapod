import numpy as np

""" Class responsible for generating Ornstein-Uhlenbeck noise"""
class Noise:
    def __init__(self, std_start,
                 std_end,
                 max_steps, std_deviation, theta, dt, decay_constant, noise_shape):
        self.std_start = std_start
        self.std_end = std_end
        self.max_steps = max_steps

        self.x_prev = None
        self.theta = theta
        self.mean = np.zeros(noise_shape)
        self.std_dev = std_start * np.ones(noise_shape)
        self.dt = dt
        self.reset_noise()
        self.decay_constant = decay_constant
        self.step = 0

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

    def decay(self):
        self.step += 1
        t = min(self.step / self.max_steps, 1.0)
        sigma = (1 - t) * self.std_start + t * self.std_end
        self.std_dev = sigma * np.ones_like(self.std_dev)

