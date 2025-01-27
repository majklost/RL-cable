import numpy as np
import pymunk

from .pm_multibody import PMMultiBodyObject
from .random_block import RandomBlock
from ..utils.seed_manager import manager


class RandomObstacleGroup(PMMultiBodyObject):
    def __init__(self, pos: np.array, VSep: int,
                 HSep: int,
                 VNum: int,
                 HNum: int,
                 btype=pymunk.Body.STATIC,
                 radius=150):
        super().__init__()
        self.VSep = VSep
        self.HSep = HSep
        self.VNum = VNum
        self.HNum = HNum
        self._btype = btype
        self._position = pos
        self.radius = radius
        self.rng = np.random.default_rng(
            manager().get_seed(self.__class__.__name__, False))
        self._create_obstacle_group()

    def _create_obstacle_group(self):
        for i in range(self.VNum):
            for j in range(self.HNum):
                r = RandomBlock(self.position + np.array(
                    [i * self.VSep, j * self.HSep]), self.radius, self._btype, seed=self.rng.integers(0, 1000))
                self.append(r)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos
