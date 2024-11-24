"""given n upper and lower bounds, sample uniformly in the n-dimensional box"""
import numpy as np

from .base_sampler import BaseSampler
from ..utils.seed_manager import manager


class NDIMSampler(BaseSampler):
    def __init__(self, lower_bounds: np.array, upper_bounds: np.array):
        super().__init__()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.ndim = len(lower_bounds)
        self.ranges = np.array(upper_bounds) - np.array(lower_bounds)
        self.rng = np.random.default_rng(
            manager().get_seed(self.__class__.__name__))

    def sample(self):
        return self.rng.random(self.ndim) * self.ranges + self.lower_bounds
