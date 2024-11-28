import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np
import gymnasium as gym
from ..sim.simulator import Simulator
from ..sim.utils.PM_debug_viewer import DebugViewer
from ..sim.objects.rectangle import Rectangle
from ..sim.utils.standard_cfg import sim_cfg


class Rectangle1D(gym.Env):
    """
    Class where there is a rectangle that must be moved to a target position.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, sim_config=sim_cfg, threshold=20, scale_factor=5000, render_mode=None, oneD=True, seed=None):
        pygame.init()
        self.seed = seed
        # rendering
        self.screen = None
        self.scale_factor = scale_factor
        self.render_mode = render_mode
        self.step_count = 0
        self.render_fps = sim_config.get("FPS", 60)

        self.width = sim_config.get("width", 800)
        self.height = sim_config.get("height", 600)

        # self.observation_space = gym.spaces.Dict({
        #     'position': gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width, self.height]), dtype=np.float64),
        #     'velocity': gym.spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float64),
        #     'target': gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width, self.height]), dtype=np.float64),
        # })
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.width, -self.height, -
                         np.inf, -np.inf], dtype=np.float32),
            high=np.array([self.width, self.height, np.inf,
                          np.inf], dtype=np.float32),
        )

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
        self.start_distance = self._calc_distance()

    # def _get_obs(self):
    #     return {
    #         'position': self.rect.position,
    #         'velocity': self.rect.velocity,
    #         'target': self.target
    #     }
    def _get_obs(self):
        return np.concatenate([self.rect.position-self.target, self.rect.velocity], dtype=np.float32)

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
        self.start_distance = self._calc_distance()
        info = self._get_info()
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        prev_distance = self._calc_distance()
        if self.oneD:
            self.rect.apply_force_middle((self.scale_factor * action, 0))
        else:
            if np.linalg.norm(action) > 1:
                action = action/np.linalg.norm(action)
            self.rect.apply_force_middle(
                self.scale_factor * action)
            # self.rect.velocity = self.scale_factor/100 * action
        self.sim.step()
        distance = self._calc_distance()
        reward = 0
        done = False
        if distance < self.threshold:
            done = True
            # reward = 500
            reward = 500-np.linalg.norm(self.rect.velocity)
        else:
            # reward = -1*distance
            reward = -5*distance/self.start_distance
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


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = Rectangle1D(oneD=False)
    check_env(env)
    gym.register(
        id='Rectangle1D-v0',
        entry_point=Rectangle1D,
    )
