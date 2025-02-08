import gymnasium as gym
import pygame.freetype
import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np
from ..sim.simulator import Simulator
from ..sim.samplers.ndim_sampler import NDIMSampler
from ..sim.objects.cable import Cable
from ..sim.maps import AlmostEmptyWorld, EmptyWorld, UPDATED_CFG, StandardStones, NonConvexWorld, ThickStones, PipedWorld


class CableRadiusEmpty(gym.Env):
    """
    Weaker version of the assignemnt where cable need to be whole in the circle around goal point
    """
    metadata = {'render.modes': ['human', None], 'render_fps': 60}

    def __init__(self, controllable_idxs=None, render_mode=None):
        super().__init__()
        pygame.init()
        self.map = self._get_map()
        # rendering
        self.screen = None
        self.width = self.map.cfg['width']
        self.height = self.map.cfg['height']
        self.render_mode = render_mode
        self.last_action = None
        self.last_reward = None
        self.cur_return = 0
        # logic
        if controllable_idxs is None:
            controllable_idxs = list(range(len(self.map.cable.bodies)))
        else:
            raise NotImplementedError("Only full control is supported")
        self.controllable_idxs = controllable_idxs
        self.controllable_num = len(controllable_idxs)
        self._set_filter()
        self.sim = self.map.get_sim()

        self.radius = self.map.cable.length / 2
        self.goal_sampler = NDIMSampler([self.radius, self.radius], [
                                        self.width - self.radius, self.height - self.radius])
        self.goal = None
        # restart
        self.map.reset_start()
        self._reset_goal()

        self.scale_factor = 600
        self.last_target_potential = 0
        self.last_info = None
        self.success = False

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

    def _reset_goal(self):
        """
        Creates 2D goal point
        """
        self.goal = self.goal_sampler.sample()

    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _create_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.controllable_num * 2,), dtype=np.float64)

    def _get_target_distance_vecs(self):
        return self.goal - self.map.cable.position

    def _get_observation(self):
        target_distance_vecs = self._get_target_distance_vecs()
        return target_distance_vecs.flatten()

    def _calc_potential(self, distances):
        return -np.sum(np.linalg.norm(distances, axis=1), where=np.linalg.norm(distances, axis=1) > self.radius)

    def _process_action(self, action):
        return action

    def step(self, action):
        action = self._process_action(action)
        self.last_action = action
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
        self.last_reward = reward

        self.last_info = self._get_info()
        return obs, reward, done, False, self.last_info

    def _get_reward(self):
        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.radius):
            self.success = True
            return 1000, True

        potential = self._calc_potential(distances)
        reward = potential - self.last_target_potential
        self.last_target_potential = potential
        return reward, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.success = False
        self.last_reward = None
        self.cur_return = 0
        self.map.reset_start()
        self._reset_goal()

        self.last_target_potential = self._calc_potential(
            self._get_target_distance_vecs())

        self.last_info = self._get_info()
        return self._get_observation(), self.last_info

    def _get_info(self):
        return {'goal': self.goal, 'last_action': self.last_action, 'success': self.success}

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
        self._render_goal(screen)
        self._render_return(screen)
        self._render_actions(screen)

    def _render_goal(self, screen):
        pygame.draw.circle(self.screen, (0, 0, 255), self.goal, 10)
        pygame.draw.circle(self.screen, (0, 0, 255), self.goal, self.radius, 1)
        target_vecs = self._get_target_distance_vecs()
        for i in range(len(target_vecs)):
            pygame.draw.line(screen, (255, 0, 0), self.map.cable.position[i],
                             self.map.cable.position[i] + target_vecs[i], 1)

    def _render_actions(self, screen):
        actions = self.last_action.reshape(
            (self.action_space.shape[0] // 2, 2))
        for i in range(len(actions)):
            pygame.draw.line(screen, (255, 0, 0), self.map.cable.bodies[i].position,
                             self.map.cable.bodies[i].position + actions[i] / 10, 2)

    def _render_return(self, screen):
        self.font.render_to(screen, (50, 50),
                            f"Reward: {self.last_reward}")
        self.font.render_to(screen, (50, 150),
                            f"Return: {self.cur_return}")

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


class CableRadiusNearestObs(CableRadiusEmpty):

    def _get_map(self):
        my_cfg = UPDATED_CFG.copy()
        my_cfg['SEG_NUM'] = 10
        return AlmostEmptyWorld(cfg=my_cfg)

    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 4,), dtype=np.float64)

    def _get_obstacle_distance_vecs(self):
        responses = np.array([self.sim._space.point_query_nearest(
            x.tolist(), (self.width**2 + self.height**2)**0.5, self.my_filter).point for x in self.map.cable.position])
        return responses - self.map.cable.position

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten()))

    def _render_obstacles_vecs(self, screen):
        obstacle_vecs = self._get_obstacle_distance_vecs()
        for i in range(len(obstacle_vecs)):
            pygame.draw.line(screen, (0, 255, 0), self.map.cable.position[i],
                             self.map.cable.position[i] + obstacle_vecs[i], 1)

    def _additional_render(self, screen):
        super()._additional_render(screen)
        self._render_obstacles_vecs(screen)

    def _get_reward(self):
        if self.map.cable.outer_collision_idxs:
            return 0, True

        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.radius):
            self.success = True
            return 1000, True

        potential = self._calc_potential(distances)
        reward = potential - self.last_target_potential
        self.last_target_potential = potential
        return reward, False


class CableRadiusNearestStronger(CableRadiusNearestObs):
    """
    Stronger reward
    """

    def _get_reward(self):
        if self.map.cable.outer_collision_idxs:
            return -500, True

        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.radius):
            self.success = True
            return 1000, True

        potential = self._calc_potential(distances)
        reward = 10 * (potential - self.last_target_potential)
        self.last_target_potential = potential
        return reward, False


class JustTest(CableRadiusNearestObs):
    def _get_map(self):
        my_cfg = UPDATED_CFG.copy()
        my_cfg['SEG_NUM'] = 10
        return StandardStones(cfg=my_cfg)


class CableRadiusObsVel(CableRadiusNearestObs):
    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 6,), dtype=np.float64)

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        velocities = self.map.cable.velocity
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten(), velocities.flatten()))


class CableRadiusObsVelStronger(CableRadiusNearestObs):
    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(low=-limit, high=limit, shape=(self.controllable_num * 6,), dtype=np.float64)

    def _get_observation(self):
        target_distances = self._get_target_distance_vecs()
        obstacle_distances = self._get_obstacle_distance_vecs()
        velocities = self.map.cable.velocity
        return np.concatenate((target_distances.flatten(), obstacle_distances.flatten(), velocities.flatten()))

    def _get_reward(self):
        if self.map.cable.outer_collision_idxs:
            return -1000, True

        distances = self._get_target_distance_vecs()
        if np.all(np.linalg.norm(distances, axis=1) < self.radius):
            self.success = True
            return 10000, True

        potential = self._calc_potential(distances)
        reward = 10 * (potential - self.last_target_potential)
        self.last_target_potential = potential
        return reward, False


class JustTestVel(CableRadiusObsVelStronger):
    def _get_map(self):
        my_cfg = UPDATED_CFG.copy()
        my_cfg['SEG_NUM'] = 10
        return PipedWorld(cfg=my_cfg)
