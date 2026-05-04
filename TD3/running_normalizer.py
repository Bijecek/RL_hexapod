import numpy as np

class RunningNormalizer:
    def __init__(self, shape):
        """
        Initialization of Welford algorithm online normalization class
        :param shape: dimension that corresponds to number of actions
        """
        self.mean = np.zeros(shape, dtype=np.float32)
        self.M2 = np.zeros(shape, dtype=np.float32)
        self.count = 0
        self.epsilon = 1e-4

    def update(self, x):
        """
        Method responsible for update of mean/variance for a single observation x
        :param x: current observation
        :return:
        """
        x = np.asarray(x, dtype=np.float32)
        self.count += 1

        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        """
        Helper method containing variance recalculation based on current entries processes
        :return:
        """
        if self.count < 2:
            return np.ones_like(self.mean)
        return self.M2 / (self.count - 1)

    def normalize(self, x):
        """
        Method handling input data normalization
        :param x: current observation
        :return: Normalized observation
        """
        return (x - self.mean) / (np.sqrt(self.variance()) + self.epsilon)

    def adjust_settings(self, normalizer_settings):
        """
        Method used for adjusting normalizer settings
        :param normalizer_settings: desired normalizer settings
        :return:
        """
        self.mean = normalizer_settings[0]
        self.M2 = normalizer_settings[1]
        self.count = normalizer_settings[2]
        self.epsilon = normalizer_settings[3]