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
        # rendering
        self.screen = None
        self.scale_factor = scale_factor
        self.render_mode = render_mode
        self.step_count = 0
        self.render_fps = sim_cfg.get("FPS", 60)

        self.width = sim_config['width']
        self.height = sim_config['height']
        # threshold for mean distance to target to consider the task solved
        self.threshold = threshold

        ctrl_num = len(self.controlable_idxs)
        limit = max(self.width, self.height)//2
        # ctrl_num*2 for distance to targets + segnum*2 for relative positions to CoG of cable
        self.observation_space = gym.spaces.Box(
            low=-limit, high=limit, shape=(ctrl_num*2+seg_num*2,), dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(ctrl_num*2,), dtype=np.float32)

        self.cable = Cable([self.width/2-cable_length/2, self.height/2],
                           length=cable_length, num_links=seg_num, thickness=5)
        self.sampler = self._create_sampler()
        self.sim = Simulator(sim_config, [self.cable], [], unstable_sim=False)
        self.exported_sim = self.sim.export()
        self.target = self._get_target()
        self.start_distance = self._calc_distance(ctrl_only=True)

    def _create_sampler(self):
        return BezierSampler(self.cable.length, self.cable.num_links, lower_bounds=np.array(
            [0, 0, np.pi]), upper_bounds=np.array([self.width, self.height, np.pi]))

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
        return self.sampler.sample(self.width/2, self.height/2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.sim.import_from(self.exported_sim)
        self.target = self._get_target()
        self.start_distance = self._calc_distance(ctrl_only=True)
        info = self._get_info()
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        # print(action)
        for i in range(len(self.controlable_idxs)):
            idx = self.controlable_idxs[i]
            # print(i, len(action), i*2+2)
            force = action[i*2:i*2+2]
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
            reward = 500
        else:
            reward = float(-5*np.mean(distance)/np.mean(self.start_distance))
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
        ctrl_num = len(self.controlable_idxs)
        limit = max(self.width, self.height)//2
        self.observation_space = gym.spaces.Box(
            low=-limit, high=limit, shape=(ctrl_num*2,), dtype=np.float32)

    def _get_obs(self):
        all_pts = self.cable.position
        ctrl_pts = all_pts[self.controlable_idxs]
        target_pts = self.target[self.controlable_idxs]
        first_seg_pt = all_pts[0]
        target_dists = target_pts - ctrl_pts
        seg_positions = all_pts - first_seg_pt
        # return np.concatenate([target_dists.flatten(), seg_positions.flatten()], dtype=np.float32
        #                       )
        return np.concatenate([target_dists.flatten()], dtype=np.float32
                              )
        # longest_dist = np.linalg.norm(
        #     np.array([self.width, self.height]))
        # normalize target points
        # target_pts /= longest_dist


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env1 = CableReshape()
    env2 = CableReshapeV2()
    check_env(env1)
    check_env(env2)
