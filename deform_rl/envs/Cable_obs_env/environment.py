import gymnasium as gym
import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np
from ..sim.simulator import Simulator
from ..sim.objects.cable import Cable
from ..sim.maps import AlmostEmptyWorld


class CableObsV0(gym.Env):
    """

    In this environment, there are a few random 
    obstacles. Agent has random initial position and random goal position.
    """
    metadata = {'render.modes': ['human', None], 'render_fps': 60}

    def __init__(self, controllable_idxs=None, threshold=20):
        super().__init__()
        pygame.init()
        self.map = AlmostEmptyWorld()
        # rendering
        self.screen = None
        self.width = self.map.cfg['width']
        self.height = self.map.cfg['height']

        # logic
        if controllable_idxs is None:
            controllable_idxs = list(range(len(self.map.cable.bodies)))
        else:
            raise NotImplementedError("Only full control is supported")
        self.controllable_idxs = controllable_idxs
        self.controllable_num = len(controllable_idxs)
        self.threshold = threshold
        self._set_filter()
        self.sim = self.map.get_sim()
        self.map.reset_goal()
        self.map.reset_start()
        self.scale_factor = 200
        self.last_target_potential = None
        self.last_obstacle_potential = None

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

    def _set_filter(self):
        for b in self.map.cable.bodies:
            for s in b.shapes:
                s.filter = pymunk.ShapeFilter(categories=0b1)
        self.my_filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)

    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 4,), dtype=np.float64)

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _get_target_distance_vecs(self):
        return self.map.cable.position - self.map.get_goal_points()

    def _get_obstacle_distance_vecs(self):
        responses = np.array([self.sim._space.point_query_nearest(
            x.tolist(), (self.width**2 + self.height**2)**0.5, self.my_filter).point for x in self.map.cable.position])
        return responses - self.map.cable.position

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten()))

    def _calc_potential(self, distances):
        return -np.sum(np.linalg.norm(distances, axis=1))

    def step(self, action):
        for i in range(len(self.controllable_idxs)):
            idx = self.controllable_idxs[i]
            # print(i, len(action), i*2+2)
            force = action[i * 2:i * 2 + 2]
            if np.linalg.norm(force) > 1:
                force /= np.linalg.norm(force)
            force *= self.scale_factor
            self.map.cable.bodies[idx].apply_force(force)

        self.sim.step()
        obs = self._get_observation()
        reward, done = self._get_reward()
        info = self._get_info()
        return obs, reward, done, False, info

    def _get_reward(self):
        if self.map.cable.outer_collision_idxs:
            return -1000, True

        if np.all(np.linalg.norm(self._get_target_distance_vecs(), axis=1) < self.threshold):
            return 1000, True

        target_potential = self._calc_potential(
            self._get_target_distance_vecs())

        obstacle_potential = self._calc_potential(
            self._get_obstacle_distance_vecs())

        target_reward = target_potential - self.last_target_potential - 5
        obstacle_reward = self.last_obstacle_potential - \
            obstacle_potential  # must be negative
        self.last_target_potential = target_potential
        self.last_obstacle_potential = obstacle_potential
        return 0.8 * target_reward + 0.2 * obstacle_reward, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.map.reset_start()
        self.map.reset_goal()
        self.last_obstacle_potential = self._calc_potential(
            self._get_obstacle_distance_vecs())
        self.last_target_potential = self._calc_potential(
            self._get_target_distance_vecs())

        info = self._get_info()
        return self._get_observation(), info

    def _get_info(self):
        return {
            "position": self.map.cable.position,
        }

    def render(self):
        if self.render_mode is None:
            return
        if self.screen is None:
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode(
                (self.width, self.height))
            self.options = DrawOptions(self.screen)
            self.options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

        self.screen.fill((255, 255, 255))
        self.sim.draw_on(self.options)
        self._additional_render(self.screen)
        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def _additional_render(self, screen):
        pass

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = CableObsV0()
    check_env(env)
