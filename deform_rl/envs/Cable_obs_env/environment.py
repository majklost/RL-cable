import gymnasium as gym
import pygame.freetype
import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np
from ..sim.simulator import Simulator
from ..sim.objects.cable import Cable
from ..sim.maps import AlmostEmptyWorld, EmptyWorld, UPDATED_CFG


class CableEmptyV0(gym.Env):
    """
    1:1 with cable movement but already with the map environment
    """
    metadata = {'render.modes': ['human', None], 'render_fps': 60}

    def __init__(self, controllable_idxs=None, threshold=20, render_mode=None):
        super().__init__()
        pygame.init()
        self.map = self._get_map()
        # rendering
        self.screen = None
        self.width = self.map.cfg['width']
        self.height = self.map.cfg['height']
        self.render_mode = render_mode

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
        self.scale_factor = 800
        # for rendering and reward calculation
        self.last_target_potential = None
        # self.last_obstacle_potential = None
        self.last_reward_obs = 0
        self.lats_reward_target = 0
        self.last_info = None
        self.last_actions = None
        self.success = False
        self.cur_return = 0

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

    def _get_map(self):
        my_cfg = UPDATED_CFG.copy()
        my_cfg['SEG_NUM'] = 10
        return EmptyWorld(cfg=my_cfg)

    def _set_filter(self):
        for b in self.map.cable.bodies:
            for s in b.shapes:
                s.filter = pymunk.ShapeFilter(categories=0b1)
        self.my_filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)

    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _get_target_distance_vecs(self):
        return self.map.get_goal_points() - self.map.cable.position

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        # obstacle_distances = self._get_obstacle_distance_vecs()
        return target_distances.flatten()

    def _calc_potential(self, distances):
        return -np.sum(np.linalg.norm(distances, axis=1))

    def step(self, action):
        self.last_actions = action
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
        self.cur_return += reward

        self.last_info = self._get_info()
        return obs, reward, done, False, self.last_info

    def _get_reward(self):
        if np.all(np.linalg.norm(self._get_target_distance_vecs(), axis=1) < self.threshold):
            self.success = True
            return 10000, True

        target_potential = self._calc_potential(
            self._get_target_distance_vecs())

        target_reward = target_potential - self.last_target_potential - 5
        self.last_target_potential = target_potential

        # DEBUG
        self.lats_reward_target = target_reward

        return target_reward, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.success = False
        self.cur_return = 0
        self.map.reset_start()
        self.map.reset_goal()
        # self.last_obstacle_potential = self._calc_potential(
        #     self._get_obstacle_distance_vecs())
        self.last_target_potential = self._calc_potential(
            self._get_target_distance_vecs())

        self.last_info = self._get_info()
        return self._get_observation(), self.last_info

    def _get_info(self):
        return {
            "position": self.map.cable.position,
            "goal": self.map.get_goal_points(),
            "success": self.success,
            # "obstacle_vecs": self._get_obstacle_distance_vecs(),
            # "target_vecs": self._get_target_distance_vecs(),
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
            pygame.font.init()
            self.font = pygame.freetype.Font('Arial.ttf', 20)
        self.screen.fill((255, 255, 255))
        self.sim.draw_on(self.options)
        self._additional_render(self.screen)
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def _additional_render(self, screen):
        self._render_gpoints(screen)
        self._render_return(screen)
        self._render_actions(screen)

    def _render_gpoints(self, screen):
        self.font.render_to(screen, (50, 100),
                            f"Target reward: {self.lats_reward_target}")
        gpoints = self.map.get_goal_points()
        for i in range(len(gpoints)):
            pygame.draw.circle(screen, (0, 255, 0), gpoints[i], 5)
        target_vecs = self._get_target_distance_vecs()
        for i in range(len(target_vecs)):
            pygame.draw.line(screen, (255, 0, 0), self.map.cable.position[i],
                             self.map.cable.position[i] + target_vecs[i], 1)

    def _render_actions(self, screen):
        actions = self.last_actions.reshape(
            (self.action_space.shape[0] // 2, 2))
        for i in range(len(actions)):
            pygame.draw.line(screen, (0, 0, 255), self.map.cable.position[i],
                             self.map.cable.position[i] + actions[i] / 10, 3)

    def _render_return(self, screen):
        self.font.render_to(screen, (50, 150),
                            f"Return: {self.cur_return}")

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


class CableEmptyV1(CableEmptyV0):
    """
    bigger observation space but part is null
    """

    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 4,), dtype=np.float64)

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        # obstacle_distances = self._get_obstacle_distance_vecs()
        return np.concatenate((target_distances.flatten(), np.zeros_like(target_distances.flatten())))


class CableEmptyV2(CableEmptyV1):
    """
    Not null observation about obstacles
    """

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten()))

    def _get_obstacle_distance_vecs(self):
        responses = np.array([self.sim._space.point_query_nearest(
            x.tolist(), (self.width**2 + self.height**2)**0.5, self.my_filter).point for x in self.map.cable.position])
        return responses - self.map.cable.position

    def _render_obstacles_vecs(self, screen):
        self.font.render_to(
            screen, (50, 50), f"Obs reward: {self.last_reward_obs}")
        obstacle_vecs = self._get_obstacle_distance_vecs()
        for i in range(len(obstacle_vecs)):
            pygame.draw.line(screen, (0, 255, 0), self.map.cable.position[i],
                             self.map.cable.position[i] + obstacle_vecs[i], 1)

    def _additional_render(self, screen):
        self._render_gpoints(screen)
        self._render_return(screen)
        self._render_actions(screen)
        self._render_obstacles_vecs(screen)
# class EmptyNoRewardObs(EmptyObsV0):
#     def _get_observation(self):
#         target_distances = self._get_target_distance_vecs()
#         obstacle_distances = np.zeros_like(target_distances)
#         return np.concatenate((target_distances.flatten(), obstacle_distances.flatten()))

#     def _get_reward(self):
#         if self.map.cable.outer_collision_idxs:
#             return -10000, True

#         if np.all(np.linalg.norm(self._get_target_distance_vecs(), axis=1) < self.threshold):
#             self.success = True
#             return 10000, True

#         target_potential = self._calc_potential(
#             self._get_target_distance_vecs())

#         target_reward = target_potential - self.last_target_potential - 5
#         self.last_target_potential = target_potential

#         # DEBUG
#         self.lats_reward_target = target_reward
#         self.last_reward_obs = 0

#         return target_reward, False


# class EmptyNoRewShorter(EmptyNoRewardObs):
#     def _get_map(self):
#         my_cfg = UPDATED_CFG.copy()
#         my_cfg['SEG_NUM'] = 10
#         return EmptyWorld(cfg=my_cfg)


# class ObsObservedOnly(CableObsV0):
#     def _get_reward(self):
#         if self.map.cable.outer_collision_idxs:
#             return -10000, True

#         if np.all(np.linalg.norm(self._get_target_distance_vecs(), axis=1) < self.threshold):
#             self.success = True
#             return 10000, True

#         target_potential = self._calc_potential(
#             self._get_target_distance_vecs())

#         target_reward = target_potential - self.last_target_potential - 5
#         self.last_target_potential = target_potential

#         # DEBUG
#         self.lats_reward_target = target_reward
#         self.last_reward_obs = 0

#         return target_reward, False
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = CableObsV0()
    check_env(env)
