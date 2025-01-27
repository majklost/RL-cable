import pymunk
import numpy as np
import random

from ..objects import *


class RandomBlock(PMSingleBodyObject):
    def __init__(self, pos: np.array, r: float, body_type, seed=None):
        super().__init__(body_type=body_type)
        self.position = pos
        self.r = r
        random.seed(seed)
        vertices = [self._get_vertex() for _ in range(8)]
        shape = pymunk.Poly(self.body, vertices)
        shape.collision_type = 2
        self.shapes.append(shape)

    def _get_vertex(self):
        r = random.random() * self.r
        # r = self.r
        theta = random.random() * 2 * 3.141592
        return r * np.cos(theta), r * np.sin(theta)
