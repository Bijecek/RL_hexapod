import numpy as np


class RunningMinMaxNormalizer:
    def __init__(self):
        self.maximum_value = float('-inf')

    def update(self, value):
        if self.maximum_value < abs(value):
            print(f"Min/Max value changed: {abs(value)}")
            self.maximum_value = abs(value)

    def normalize(self, x):
        if abs(x) > self.maximum_value:
            x = float(np.clip(x, -self.maximum_value, self.maximum_value))

        return 2 * ( (x - -self.maximum_value) / (self.maximum_value - -self.maximum_value) ) - 1

    def normalize_restricted(self, x):
        if abs(x) > self.maximum_value:
            x = float(np.clip(x, -self.maximum_value, self.maximum_value))

        return (x + self.maximum_value) / (2 * self.maximum_value)