import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from ..sim.simulator import Simulator
from ..sim.utils.PM_debug_viewer import DebugViewer
from ..sim.objects.rectangle import Rectangle
from ..sim.utils.standard_cfg import sim_cfg


class BaseRectangleEnv(gym.Env):
    """
    Class where there is a rectangle that must be moved to a target position.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sim_config=sim_cfg, threshold=15, scale_factor=5000, max_steps=1000, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.threshold = threshold
        self.screen = None
        self.launch_cnt = 0  # for tweaking random sampling
        self.scale_factor = scale_factor
        self.step_count = 0
        self.fps = sim_config['FPS']
        self.width, self.height = sim_cfg['width'], sim_cfg['height']

        self.max_steps = max_steps

        self.observation_space = self._get_obs_space()
        self.action_space = self._get_action_space()

        self.rect = Rectangle([self.width/2, self.height/2],
                              20, 20, pymunk.Body.DYNAMIC)

        self.sim = Simulator(sim_cfg, [self.rect], [], unstable_sim=False)
        self.exported_sim = self.sim.export()
        self.target = self._get_target()
        self.last_action = None
        self.previous_position = None

    def _get_obs_space(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _get_target(self):
        raise NotImplementedError()

    def _get_position(self):
        raise NotImplementedError()

    def _standard_reward(self):
        raise NotImplementedError()

    def _get_action_space(self):
        limit = np.array([1, 1])
        return gym.spaces.Box(-limit, limit, dtype=np.float32)

    def _action_modifier(self, action):
        return self.scale_factor * action

    def _calc_distance(self):
        return np.linalg.norm(self.rect.position - self.target)

    def _on_finish_reward(self):
        return 1000

    def _get_info(self):
        return {
            "distance": self._calc_distance(),
            "position": self.rect.position,
            "velocity": self.rect.velocity,
            "last_action": self.last_action
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.step_count = 0
        self.sim.import_from(self.exported_sim)
        self.rect.position = self._get_position()
        self.previous_position = self.rect.position
        self.target = self._get_target()
        info = self._get_info()

        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        action = self._action_modifier(action)
        self.last_action = action
        self.previous_position = self.rect.position
        self.rect.apply_force_middle(action)
        self.sim.step()
        reward = 0
        done = False
        distance = self._calc_distance()
        if distance < self.threshold:
            done = True
            reward += self._on_finish_reward()

        # truncated = self.step_count >= self.max_steps

        reward += self._standard_reward()
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
        self.clock.tick(self.fps)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


class RectNoVel(BaseRectangleEnv):
    def __init__(self, sim_config=sim_cfg, threshold=15, scale_factor=5000, max_steps=1000, gamma=0.99, render_mode=None):
        super().__init__(sim_config, threshold, scale_factor, max_steps, render_mode)
        self.gamma = gamma

    def _get_obs_space(self):
        limit = np.array([np.inf, np.inf])
        return gym.spaces.Box(-limit, limit, dtype=np.float32)

    def _get_obs(self):
        return (self.target-self.rect.position).astype(np.float32)

    def _get_target(self):
        PADDING = 20
        return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])

    def _get_position(self):
        PADDING = 20
        return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])

    def calc_potential(self, position):
        # max_distance = np.sqrt(self.width**2 + self.height**2)
        return -np.linalg.norm(position-self.target)

    def _standard_reward(self):
        prev_potential = self.calc_potential(self.previous_position)
        now_potential = self.calc_potential(self.rect.position)
        # result = self.gamma * now_potential - prev_potential
        result = now_potential - prev_potential-5
        # print("Giving reward: ", result)
        return result
        # distance = self._calc_distance()
        # return -5*distance-10

    def _action_modifier(self, action):
        if np.linalg.norm(action) > 1:
            action = action/np.linalg.norm(action)
        return self.scale_factor * action


class RectVel(RectNoVel):
    def _get_obs_space(self):

        limit = np.array([np.inf, np.inf, np.inf, np.inf])
        return gym.spaces.Box(-limit, limit, dtype=np.float32)

    def _get_obs(self):
        # print("OBS")
        return np.concatenate((self.target-self.rect.position, self.rect.velocity), dtype=np.float32)


class TrajectoryVel(RectVel):
    def _get_target(self):
        PADDING = 20
        return np.array([self.width, self.height])/2

    def _get_position(self):
        PADDING = 20
        return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])


class TrajectoryNoVel(RectNoVel):
    def _get_target(self):
        PADDING = 20
        return np.array([self.width, self.height])/2

    def _get_position(self):
        PADDING = 20
        return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])


class Circles(RectVel):
    def __init__(self, sim_config=sim_cfg, threshold=15, scale_factor=5000, max_steps=1000,  render_mode=None, num_in_circle=60, radius=100, gamma=0.99):
        super().__init__(sim_config, threshold, scale_factor,
                         max_steps, render_mode=render_mode)
        self.num_in_circle = num_in_circle
        self.cur_circle = 0
        self.radius = radius

    def _get_target(self):
        PADDING = 20
        return np.array([self.width, self.height])/2

    def _get_position(self):
        PADDING = 20
        if self.cur_circle == self.num_in_circle:
            self.radius *= 1.5
            self.cur_circle = 0
        angle = (2*np.pi*self.cur_circle+10*self.radius)/self.num_in_circle
        self.cur_circle += 1
        return np.array([self.width/2+self.radius*np.cos(angle), self.height/2+self.radius*np.sin(angle)])


if __name__ == '__main__':
    check_env(RectNoVel())
    check_env(RectVel())
