import pymunk
import pygame
import numpy as np
import gymnasium as gym
from .simulator import Simulator
from .utils.PM_debug_viewer import DebugViewer
from .objects.rectangle import Rectangle


class Rectangle(gym.Env):
    """
    Class where there is a rectangle that must be moved to a target position.

    """

    def __init__(self, config):
        self.width = config.get("width", 800)
        self.height = config.get("height", 600)
        self.observation_space = gym.spaces.Box([0]*6,[self.width,self.height]*3)

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def _get_reward(self):
        pass
