import numpy as np

class Noise:
    def __init__(self, std_start,
                 std_end,
                 max_steps, theta, dt, noise_shape, action_range):
        """
        Class responsible for generating Ornstein-Uhlenbeck noise
        :param std_start: constant representing maximum value of standard deviation
        :param std_end: constant representing minimum value of standard deviation
        :param max_steps: constant representing maximum number of steps used in linear decay
        :param theta: constant of OU noise setting
        :param dt: constant of OU noise setting
        :param noise_shape: dimension that corresponds to number of actions
        :param action_range: action range of said actions
        """
        self.std_start = std_start
        self.std_end = std_end
        self.max_steps = max_steps
        self.action_range = np.array(action_range)

        self.noise_shape = noise_shape
        self.x_prev = None
        self.theta = theta
        self.mean = np.zeros(noise_shape)
        self.std_dev = std_start * np.ones(noise_shape)
        self.dt = dt
        self.reset_noise()
        self.step = 0

    def __call__(self):
        """
        Method responsible for generating Ornstein-Uhlenbeck noise
        :return: Scaled value of noise to match action range values
        """
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        """ Make the next noise depend on the current one"""
        self.x_prev = x
        return x * self.action_range

    def reset_noise(self):
        self.x_prev = np.zeros_like(self.mean)

    def decay(self):
        """
        Linear decay implementation which over time lowers standard deviation of this noise (depending on number of steps of simulation)
        :return:
        """
        self.step += 1
        t = min(self.step / self.max_steps, 1.0)
        sigma = (1 - t) * self.std_start + t * self.std_end
        self.std_dev = sigma * np.ones_like(self.std_dev)

