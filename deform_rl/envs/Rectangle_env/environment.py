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
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sim_config=sim_cfg, threshold=20, scale_factor=5000, render_mode=None, oneD=False):
        pygame.init()
        # rendering
        self.screen = None
        self.launch_cnt = 0
        self.scale_factor = scale_factor
        self.render_mode = render_mode
        self.step_count = 0
        self.render_fps = sim_config.get("FPS", 60)

        self.width = sim_config.get("width", 800)
        self.height = sim_config.get("height", 800)

        self.observation_space = self._get_obs_space()

        self.threshold = threshold

        self.oneD = oneD
        if oneD:
            self.action_space = gym.spaces.Box(low=np.array(
                [-1]), high=np.array([1]), dtype=np.float32)
        else:
            self.action_space = self._get_action_space()

        self.rect = Rectangle([self.width/2, self.height/2],
                              20, 20, pymunk.Body.DYNAMIC)
        self.sim_cfg = sim_config
        self.sim = Simulator(
            self.sim_cfg, [self.rect], [], unstable_sim=False)
        self.exported_sim = self.sim.export()
        self.target = self._get_target()
        self.start_distance = self._calc_distance()

    def _get_obs_space(self):
        return gym.spaces.Box(
            low=np.array([-self.width, -self.height, -
                         np.inf, -np.inf], dtype=np.float32),
            high=np.array([self.width, self.height, np.inf,
                          np.inf], dtype=np.float32),
        )

    def _get_action_space(self):
        return gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
        )

    def _get_obs(self):
        return np.concatenate([self.rect.position-self.target, self.rect.velocity], dtype=np.float32)

    def _get_info(self):
        return {
            "distance": self._calc_distance(),
            "position": self.rect.position,
        }

    def _calc_distance(self):
        return np.linalg.norm(self.rect.position - self.target)

    def _get_target(self):
        if self.oneD:
            return np.array([self.np_random.integers(0, self.width), self.height/2])
        else:
            PADDING = 20
            return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])

    def _get_position(self):
        return self._get_target()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.step_count = 0
        self.sim.import_from(self.exported_sim)
        self.rect.position = self._get_position()
        self.target = self._get_target()
        self.start_distance = self._calc_distance()
        info = self._get_info()
        return self._get_obs(), info

    def _action_modifier(self, action):
        if np.linalg.norm(action) > 1:
            action = action/np.linalg.norm(action)
        return self.scale_factor * action

    def step(self, action):
        self.step_count += 1
        prev_distance = self._calc_distance()
        if self.oneD:
            self.rect.apply_force_middle((self.scale_factor * action, 0))
        else:
            action = self._action_modifier(action)
            self.rect.apply_force_middle(
                action)
            # self.rect.velocity = self.scale_factor/100 * action
        self.sim.step()
        distance = self._calc_distance()
        reward = 0
        done = False
        if distance < self.threshold:
            done = True
            reward = self._on_finish_reward()
            # reward = 500-np.linalg.norm(self.rect.velocity)
        elif self.rect.position[0] < 0 or self.rect.position[0] > self.width or self.rect.position[1] < 0 or self.rect.position[1] > self.height:
            done = True
            reward = -500
        else:
            # reward = -1*distance
            reward = -5*distance/self.start_distance-10
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info

    def _on_finish_reward(self):
        return 1000

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


class RectangleNoVel(Rectangle1D):
    """
    Class where No velocity is observed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_obs_space(self):
        return gym.spaces.Box(
            low=np.array([-self.width, -self.height], dtype=np.float32),
            high=np.array([self.width, self.height], dtype=np.float32),
        )

    def _get_obs(self):
        return np.concatenate([self.rect.position-self.target], dtype=np.float32)


class RectangleVelDirOnly(Rectangle1D):
    """
    """

    def _get_obs_space(self):
        return gym.spaces.Box(
            low=np.array([-self.width, -self.height, -1, -1],
                         dtype=np.float32),
            high=np.array([self.width, self.height, 1, 1], dtype=np.float32),
        )

    def _get_obs(self):
        return np.concatenate([self.rect.position-self.target, self.rect.velocity/(np.linalg.norm(self.rect.velocity)+1e-6)], dtype=np.float32)


class RectangleVelReward(Rectangle1D):
    """
    Penalizes velocity in finish.
    """

    def _on_finish_reward(self):
        return 500-3*np.linalg.norm(self.rect.velocity)


class RenderingEnv(Rectangle1D):

    def _get_position(self):
        PADDING = 20
        # print(self.launch_cnt)
        # return np.array([41, 783])
        # if self.launch_cnt == 0:
        #     self.launch_cnt += 1
        #     return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])
        # elif self.launch_cnt == 1:
        #     self.launch_cnt += 1
        #     return np.array([301, 429])
        # elif self.launch_cnt == 2:
        #     self.launch_cnt += 1
        #     return np.array([301, 429])
        # elif self.launch_cnt == 3:
        #     self.launch_cnt += 1
        #     return np.array([41, 783])
        #     # return np.array([200, 200])
        # elif self.launch_cnt == 4:
        #     self.launch_cnt += 1
        #     return np.array([41, 783])
        #     # return np.array([200, 200])
        # else:
        #     raise ValueError()
        return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])

    def _get_target(self):
        return np.array([self.width/2, self.height/2])


class RenderingEnvVelocity(RectangleNoVel):
    """
    Movement is created not be force but by velocity
    """

    def _get_position(self):
        PADDING = 20
        return np.array([self.np_random.integers(PADDING, self.width-PADDING), self.np_random.integers(PADDING, self.height-PADDING)])

    def _get_target(self):
        return np.array([self.width/2, self.height/2])

    def _action_modifier(self, action):
        if np.linalg.norm(action) > 1:
            action = action/np.linalg.norm(action)
        return action

    def step(self, action):
        self.step_count += 1
        action = self._action_modifier(action)
        # self.rect.apply_force_middle(
        #     action)
        self.rect.velocity = 1500*action
        print(action)
        self.sim.step()
        distance = self._calc_distance()
        reward = 0
        done = False
        if distance < self.threshold:
            done = True
            reward = self._on_finish_reward()
            # reward = 500-np.linalg.norm(self.rect.velocity)
        elif self.rect.position[0] < 0 or self.rect.position[0] > self.width or self.rect.position[1] < 0 or self.rect.position[1] > self.height:
            done = True
            reward = -500
        else:
            # reward = -1*distance
            reward = -5*distance/self.start_distance-10
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info


class RectPolar(Rectangle1D):
    """
    Action space is polar coordinates.
    """

    def _get_action_space(self):
        # first is R and second is theta
        return gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
        )

    def _action_modifier(self, action):
        r = (action[0]+1) * self.scale_factor
        theta = action[1] * np.pi
        return np.array([r*np.cos(theta), r*np.sin(theta)])


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = Rectangle1D(oneD=False)
    check_env(env)
    gym.register(
        id='Rectangle1D-v0',
        entry_point=Rectangle1D,
    )
