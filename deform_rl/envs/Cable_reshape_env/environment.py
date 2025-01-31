import gymnasium as gym
import pymunk
from pymunk.pygame_util import DrawOptions
import pygame
import numpy as np
from ..sim.simulator import Simulator
from ..sim.objects.cable import Cable
from ..sim.utils.PM_debug_viewer import DebugViewer
from ..sim.utils.standard_cfg import sim_cfg
from ..sim.samplers.bezier_sampler import BezierSampler


class CableReshape(gym.Env):
    """

    Class where cable is spawned in the middle of the screen and must be reshaped to a target shape.
    Target shape is also in the middle of the screen.
    Observations are already normalized.
    """
    metadata = {'render.modes': ['human', None], 'render_fps': 60}

    def __init__(self, sim_config=sim_cfg, threshold=20, seg_num=5, controlable_idxs=None,
                 cable_length=300, scale_factor=200, render_mode=None, seed=None):
        pygame.init()
        if controlable_idxs is None:
            controlable_idxs = list(range(seg_num))
        self.controlable_idxs = np.array(controlable_idxs)
        self.seed = seed
        self.prev_points = None
        # rendering
        self.seg_num = seg_num
        self.screen = None
        self.scale_factor = scale_factor
        self.render_mode = render_mode
        self.step_count = 0
        self.render_fps = sim_cfg.get("FPS", 60)

        self.width = sim_config['width']
        self.height = sim_config['height']
        # threshold for mean distance to target to consider the task solved
        self.threshold = threshold

        self.ctrl_num = len(self.controlable_idxs)
        # ctrl_num*2 for distance to targets + segnum*2 for relative positions to CoG of cable
        self.observation_space = self._create_observation_space()

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.ctrl_num * 2,), dtype=np.float32)

        self.cable = Cable([self.width / 2 - cable_length / 2, self.height / 2],
                           length=cable_length, num_links=seg_num, thickness=5)
        self.sampler = self._create_sampler()
        self.sim = Simulator(sim_config, [self.cable], [], unstable_sim=False)
        self.exported_sim = self.sim.export()
        self.target = self._get_target()
        self.start_distance = self._calc_distance(ctrl_only=True)

    def _create_observation_space(self):
        ctrl_num = len(self.controlable_idxs)
        limit = max(self.width, self.height)
        return gym.spaces.Box(
            low=-limit, high=limit, shape=(ctrl_num * 2 + self.seg_num * 2,), dtype=np.float32)

    def _create_sampler(self):
        PADDING = 50
        return BezierSampler(self.cable.length, self.cable.num_links, lower_bounds=np.array(
            [PADDING, PADDING, 0]), upper_bounds=np.array([self.width - PADDING, self.height - PADDING, 2 * np.pi]))

    def _calc_distance(self, ctrl_only=False):
        # ctrl_points are in shape (num_links, 2)
        if ctrl_only:
            ctrl_points = self.cable.position[self.controlable_idxs]
            target_points = self.target[self.controlable_idxs]
        else:
            ctrl_points = self.cable.position
            target_points = self.target

        # return each distance between control points and target points
        return np.linalg.norm(ctrl_points - target_points, axis=1)

    def _get_obs(self):
        all_pts = self.cable.position
        ctrl_pts = all_pts[self.controlable_idxs]
        target_pts = self.target[self.controlable_idxs]
        cog = np.mean(all_pts, axis=0)
        # normalize rel_pts to be in range [-1, 1]
        rel_pts = all_pts - cog
        rel_pts_norm_coef = np.max(np.linalg.norm(rel_pts, axis=1))
        rel_pts /= rel_pts_norm_coef

        target_dist = (target_pts - ctrl_pts) / \
            np.array([self.width, self.height])
        return np.concatenate((target_dist.flatten(), rel_pts.flatten()), dtype=np.float32)

    def _get_info(self):
        return {
            "distance": self._calc_distance(),
            "target": self.target,
            "position": self.cable.position,
        }

    def _get_target(self):
        return self.sampler.sample(self.width / 2, self.height / 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.sim.import_from(self.exported_sim)
        self.target = self._get_target()
        self.start_distance = self._calc_distance(ctrl_only=True)
        info = self._get_info()
        return self._get_obs(), info

    def _get_standard_reward(self):
        distance = self._calc_distance(ctrl_only=True)
        reward = float(-5 * np.mean(distance) / np.mean(self.start_distance))
        return reward

    def step(self, action):
        self.step_count += 1
        self.prev_points = self.cable.position
        # print(action)
        for i in range(len(self.controlable_idxs)):
            idx = self.controlable_idxs[i]
            # print(i, len(action), i*2+2)
            force = action[i * 2:i * 2 + 2]
            if np.linalg.norm(force) > 1:
                force /= np.linalg.norm(force)
            force *= self.scale_factor
            self.cable.bodies[idx].apply_force_middle(force)

        self.sim.step()
        distance = self._calc_distance(ctrl_only=True)
        reward = 0
        done = False
        if np.all(distance < self.threshold):
            done = True
            reward = 1000

        reward += self._get_standard_reward()
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info

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
        self._render_target(self.screen)
        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def _render_target(self, screen):
        for i in range(len(self.target)):
            pygame.draw.circle(screen, (0, 255, 0), self.target[i], 5)
        for i in range(len(self.target)):
            pygame.draw.line(screen, (255, 0, 0), self.cable.position[i],
                             self.target[i], 1)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


class CableReshapeV2(CableReshape):
    """
    Cable reshape with only distance to target points as observation.
    """

    def __init__(self, sim_config=sim_cfg, threshold=20, seg_num=5, controlable_idxs=None,
                 cable_length=300, scale_factor=200, render_mode=None, seed=None):
        super().__init__(sim_config, threshold, seg_num, controlable_idxs,
                         cable_length, scale_factor, render_mode, seed)

        self.observation_space = self._create_observation_space()

    def _create_observation_space(self):
        ctrl_num = len(self.controlable_idxs)
        limit = max(self.width, self.height) // 2
        return gym.spaces.Box(
            low=-limit, high=limit, shape=(ctrl_num * 2,), dtype=np.float32)

    def _get_obs(self):
        all_pts = self.cable.position
        ctrl_pts = all_pts[self.controlable_idxs]
        target_pts = self.target[self.controlable_idxs]
        first_seg_pt = all_pts[0]
        target_dists = target_pts - ctrl_pts
        seg_positions = all_pts - first_seg_pt
        # print(ctrl_pts.shape)
        # return np.concatenate([target_dists.flatten(), seg_positions.flatten()], dtype=np.float32
        #                       )
        return np.concatenate([target_dists.flatten()], dtype=np.float32
                              )
        # longest_dist = np.linalg.norm(
        #     np.array([self.width, self.height]))
        # normalize target points
        # target_pts /= longest_dist


class CableReshapeV3(CableReshapeV2):
    """
    Cable with reward shaping inspired by debug learn
    """

    def calc_potential(self, position):
        target_pts = self.target[self.controlable_idxs]
        # max_distance = np.sqrt(self.width**2 + self.height**2)
        return -np.sum(np.linalg.norm(position - self.target, axis=1))

    def _get_standard_reward(self):
        all_pts = self.cable.position
        ctrl_pts = all_pts[self.controlable_idxs]
        prev_pts = self.prev_points[self.controlable_idxs]
        prev_potential = self.calc_potential(prev_pts)
        now_potential = self.calc_potential(ctrl_pts)
        return now_potential - prev_potential - 5


class CableReshapeMovement(CableReshapeV3):
    def _get_target(self):
        return self.sampler.sample()


class CableReshapeMovementVel(CableReshapeMovement):
    def _create_observation_space(self):
        ctrl_num = len(self.controlable_idxs)
        limit = np.inf

        return gym.spaces.Box(
            low=-limit, high=limit, shape=(ctrl_num * 4,), dtype=np.float32)

    def _get_obs(self):
        all_pts = self.cable.position
        ctrl_pts = all_pts[self.controlable_idxs]
        target_pts = self.target[self.controlable_idxs]
        target_dists = target_pts - ctrl_pts
        vel = self.cable.velocity[self.controlable_idxs]
        return np.concatenate([target_dists.flatten(), vel.flatten()], dtype=np.float32
                              )


class CableReshapeMovementNeighbourObs(CableReshapeMovement):
    def _create_observation_space(self):
        ctrl_num = len(self.controlable_idxs)
        limit = np.inf

        return gym.spaces.Box(
            low=-limit, high=limit, shape=(ctrl_num * 6,), dtype=np.float32)

    def _get_obs(self):
        all_pts = self.cable.position
        ctrl_pts = all_pts[self.controlable_idxs]
        target_pts = self.target[self.controlable_idxs]
        target_dists = target_pts - ctrl_pts
        next_pts_idx = (np.arange(self.ctrl_num) + 1) % self.ctrl_num
        next_pts = ctrl_pts[next_pts_idx]
        rel_pts = next_pts - all_pts
        vel = self.cable.velocity[self.controlable_idxs]
        return np.concatenate([target_dists.flatten(), rel_pts.flatten(), vel.flatten()], dtype=np.float32)


class CableReshapeHardFlips(CableReshapeV2):

    def _create_sampler(self):
        return BezierSampler(self.cable.length, self.cable.num_links, lower_bounds=np.array(
            [0, 0, np.pi]), upper_bounds=np.array([self.width, self.height, np.pi]))


class CableReshapeMovementOld(CableReshapeV2):
    def _get_target(self):
        return self.sampler.sample()


class CableReshapeNeighbourObs(CableReshapeV2):
    """
    Cable reshape distance to target and relative position of neighbours as observation.
    """

    def __init__(self, sim_config=sim_cfg, threshold=20, seg_num=5, controlable_idxs=None, cable_length=300, scale_factor=200, render_mode=None, seed=None):
        super().__init__(sim_config, threshold, seg_num, controlable_idxs,
                         cable_length, scale_factor, render_mode, seed)

    def _create_observation_space(self):
        limit = max(self.width, self.height)
        return gym.spaces.Box(
            low=-limit, high=limit, shape=((self.ctrl_num + self.ctrl_num) * 2,), dtype=np.float32)

    def _get_obs(self):
        all_pts = self.cable.position
        ctrl_pts = all_pts[self.controlable_idxs]
        target_pts = self.target[self.controlable_idxs]
        target_dists = target_pts - ctrl_pts
        next_pts_idx = (np.arange(self.ctrl_num) + 1) % self.ctrl_num
        next_pts = ctrl_pts[next_pts_idx]
        rel_pts = next_pts - all_pts
        return np.concatenate([target_dists.flatten(), rel_pts.flatten()], dtype=np.float32)

        # for i in range(self.seg_num):
        #     next_pts_idx = (i+1) % self.seg_num
        #     next_pts_pos = all_pts[next_pts_idx]
        #     cur_pts_pos = all_pts[i]


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env1 = CableReshape()
    env2 = CableReshapeV2()
    env3 = CableReshapeHardFlips()
    env4 = CableReshapeMovement()  # TODO: fix this after hyperopt
    env5 = CableReshapeNeighbourObs()
    check_env(env1)
    check_env(env2)
    check_env(env3)
    check_env(env4)
    check_env(env5)
