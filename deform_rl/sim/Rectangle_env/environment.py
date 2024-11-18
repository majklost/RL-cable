import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np
import gymnasium as gym
from .simulator import Simulator
from .utils.PM_debug_viewer import DebugViewer
from .objects.rectangle import Rectangle


class Rectangle1D(gym.Env):
    """
    Class where there is a rectangle that must be moved to a target position.
    """
    metadata = {'render.modes': ['human', None], 'render_fps': 60}

    def __init__(self, sim_config, threshold=2, scale_factor=5000, render_mode=None, oneD=True, seed=None):
        pygame.init()
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # rendering
        self.screen = None
        self.scale_factor = scale_factor
        self.render_mode = render_mode
        self.step_count = 0
        self.render_fps = sim_config.get("FPS", 60)

        self.width = sim_config.get("width", 800)
        self.height = sim_config.get("height", 600)

        self.observation_space = gym.spaces.Dict({
            'position': gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width, self.height]), dtype=np.float64),
            'velocity': gym.spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float64),
            'target': gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width, self.height]), dtype=np.float64),
        })
        self.threshold = threshold

        self.oneD = oneD
        if oneD:
            self.action_space = gym.spaces.Box(low=np.array(
                [-1]), high=np.array([1]), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=np.array(
                [-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        self.rect = Rectangle([self.width/2, self.height/2],
                              20, 20, pymunk.Body.DYNAMIC)
        self.sim_cfg = sim_config
        self.sim = Simulator(
            self.sim_cfg, [self.rect], [], unstable_sim=False)
        self.exported_sim = self.sim.export()
        self.target = self._get_target()

    def _get_obs(self):
        return {
            'position': self.rect.position,
            'velocity': self.rect.velocity,
            'target': self.target
        }

    def _get_info(self):
        return {
            "distance": self._calc_distance(),
        }

    def _calc_distance(self):
        return np.linalg.norm(self.rect.position - self.target)

    def _get_target(self):
        if self.oneD:
            return np.array([self.np_random.integers(0, self.width), self.height/2])
        else:
            return np.array([self.np_random.integers(0, self.width), self.np_random.integers(0, self.height)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.sim.import_from(self.exported_sim)
        self.rect.position = self._get_target()
        self.target = self._get_target()
        info = self._get_info()
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        prev_distance = self._calc_distance()
        if self.oneD:
            self.rect.apply_force_middle((self.scale_factor * action, 0))
        else:
            self.rect.apply_force_middle(self.scale_factor * action)
        self.sim.step()
        distance = self._calc_distance()
        reward = 0
        done = False
        if distance < self.threshold:
            done = True
            velocity = np.linalg.norm(self.rect.velocity)
            reward = 10*(1000 - self.step_count)
            # print("Reached target in {} steps with velocity {}".format(
            #     self.step_count, velocity))
        else:
            reward = prev_distance-distance
        # reward -= (.01*self.step_count)
        # reward -= 10
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info

    def render(self):
        if self.render_mode != 'human':
            return
        if self.screen is None:
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.draw_options = DrawOptions(self.screen)
        self.screen.fill((255, 255, 255))
        self.sim.draw_on(self.draw_options)
        pygame.draw.circle(self.screen, (255, 0, 0),
                           self.target, 10, 0)
        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
