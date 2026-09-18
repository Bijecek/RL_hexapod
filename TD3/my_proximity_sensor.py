import numpy as np
from pyrep.objects import ProximitySensor
from math import sqrt
from pyrep.backend import sim

class MyProximitySensor(ProximitySensor):
    def read(self):
        """
        Wrapper for sim.simReadProximitySensor() method which adds distance calculation
        :return: Boolean flag of measurement, distance, np.array(point), np.array(normal)
        """
        state, _, point, normal = sim.simReadProximitySensor(self._handle)

        if state:
            distance = sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
            return True, distance, np.array(point), np.array(normal)

        return False, -1.0, None, None
