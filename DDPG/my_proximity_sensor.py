import numpy as np
from pyrep.objects import ProximitySensor
from pyrep.objects.object import Object
from pyrep.const import ObjectType
from math import sqrt
from pyrep.backend import sim

class MyProximitySensor(ProximitySensor):
    def read(self):
        state, _, point, normal = sim.simReadProximitySensor(self._handle)

        if state:
            distance = sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
            return True, distance, np.array(point), np.array(normal)

        return False, -1.0, None, None