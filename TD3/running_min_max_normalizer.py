import numpy as np

class RunningMinMaxNormalizer:
    def __init__(self):
        """
        Initialization of min-max normalization class
        """
        self.maximum_value = float('-inf')

    def update(self, value):
        """
        Method responsible for updating maximum value
        :param value: current value
        :return:
        """
        if self.maximum_value < abs(value):
            self.maximum_value = abs(value)

    def normalize(self, x):
        """
        Implementation of min-max normalization logic
        :param x: input value
        :return:
        """
        if abs(x) > self.maximum_value:
            x = float(np.clip(x, -self.maximum_value, self.maximum_value))

        return 2 * ( (x - -self.maximum_value) / (self.maximum_value - -self.maximum_value) ) - 1