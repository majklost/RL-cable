import numpy as np

from .objects import *
from .utils.PM_rectangle_controller import PMRectangleController
# from  import ConfigManager
from .utils.seed_manager import init_manager
from .simulator import Simulator
# from deform_plan.utils.PM_space_visu import show_sim
from .utils.PM_debug_viewer import DebugViewer
from .samplers.bezier_sampler import BezierSampler
from .samplers.ndim_sampler import NDIMSampler
from pymunk import Body


# suggested config for these maps
CABLE_LENGTH = 300
WIDTH = 1900
HEIGHT = 800

CFG = {
    'width': WIDTH,
    'height': HEIGHT,
    'FPS': 60,
    'gravity': 0,
    'damping': .15,
    'collision_slope': 0.01,
}


# plan - 320 px start, 320 px end; middle is obstacle course
EMPTY = 320
START = (EMPTY - 10, HEIGHT // 2 - CABLE_LENGTH // 2)
END = (WIDTH - EMPTY // 2, HEIGHT // 2)


def deg2rad(deg):
    return deg * np.pi / 180


class EmptyWorld:
    """
    Empty world
    """

    def __init__(self, movable_obj_maker, sampler, cfg=CFG, seed_env=None):
        self.cfg = cfg

        init_manager(seed_env, None)
        self.fixed = []
        self.movable = []
        self._add_basic()
        self._add_obj(movable_obj_maker)

        self._sampler = sampler
        self._in_cnt = 0

    @staticmethod
    def get_movable_idx():
        return 0

    def _begin_col(self, arbiter, space, data):
        self._in_cnt += 1
        print("Collision")
        return False

    def _on_exit(self, arbiter, space, data):
        self._in_cnt -= 1
        return False

    def _add_basic(self):
        boundings = Boundings(self.cfg["width"], self.cfg["height"])
        end_platform = Rectangle(
            np.array([WIDTH - EMPTY // 2 - 10, HEIGHT // 2]), EMPTY, HEIGHT, STATIC, False)
        end_platform.color = (237, 186, 31)
        end_platform.set_collision_type(5)

        self.fixed.append(boundings)
        self.fixed.append(end_platform)

    def _add_obj(self, maker):
        self.obj = Rectangle(START, 20, 20, DYNAMIC)
        self.obj.set_collision_type(1)
        self.obj.position = self.get_start()
        self.movable.append(self.obj)

    @staticmethod
    def get_start():
        return START

    @staticmethod
    def get_goal():
        return END

    def get_sim(self):
        sim = Simulator(self.cfg, self.movable, self.fixed,
                        threaded=False, unstable_sim=False)
        sim.add_custom_handler(self._begin_col, self._on_exit, 1, 5)
        return sim


class PipedWorld(EmptyWorld):
    """
    World with narrow passages between 'rooms' some feasible, some not
    """

    def __init__(self, obj_maker, sampler, cfg=CFG, seed_env=None):

        super().__init__(obj_maker, sampler, cfg, seed_env)
        self._add_pipes()

    def _add_pipes(self):
        blockage = Rectangle(
            np.array([EMPTY + 50, HEIGHT // 4 - 50]), HEIGHT // 2 - 80, 20, STATIC)
        blockage2 = Rectangle(np.array(
            [EMPTY + 50, HEIGHT - (HEIGHT // 4 - 50)]), HEIGHT // 2 - 80, 20, STATIC)
        blockage.orientation = blockage2.orientation = np.pi / 2
        self.fixed.append(blockage)
        self.fixed.append(blockage2)
        self._create_pipe(np.array([EMPTY + 200, HEIGHT // 2]), 300, 0, 150)
        self._add_v_shape(np.array([WIDTH // 2 - 150, HEIGHT // 2]), 200, 120)
        rec = Rectangle(
            np.array([WIDTH // 2 - 220, HEIGHT // 2 - 200]), 250, 20, STATIC)
        rec.orientation = -deg2rad(60)
        self.fixed.append(rec)
        rec = Rectangle(
            np.array([WIDTH // 2 - 220, HEIGHT // 2 + 180]), 250, 20, STATIC)
        rec.orientation = deg2rad(45)
        self.fixed.append(rec)
        rec = Rectangle(
            np.array([WIDTH // 2, HEIGHT // 2 + 280]), 400, 20, STATIC)
        self.fixed.append(rec)
        rec = Rectangle(
            np.array([WIDTH // 2, HEIGHT // 2 + 200]), 400, 20, STATIC)
        rec.orientation = deg2rad(45)
        self.fixed.append(rec)
        rec = Rectangle(
            np.array([WIDTH // 2 + 80, HEIGHT // 2 - 280]), 700, 20, STATIC)
        self.fixed.append(rec)
        rec = Rectangle(
            np.array([WIDTH // 2 + 20, HEIGHT // 2 - 120]), 300, 20, STATIC)
        self.fixed.append(rec)
        rec = Rectangle(
            np.array([WIDTH // 2 + 400, HEIGHT // 2 - 120]), 400, 20, STATIC)
        rec.orientation = -deg2rad(70)
        self.fixed.append(rec)
        # self._create_pipe(np.array([EMPTY+500, HEIGHT // 2+180]), 300, deg2rad(45), 150)

    def _create_pipe(self, pos, length, angle, width):
        pipe1 = Rectangle(pos + (0, width // 2), length, 20, STATIC)
        pipe1.orientation = angle
        pipe2 = Rectangle(pos - (0, width // 2), length, 20, STATIC)
        pipe2.orientation = angle
        self.fixed.append(pipe1)
        self.fixed.append(pipe2)

    def _add_v_shape(self, pos, length, angle):
        rect1 = Rectangle(pos + (0, length // 3), length, 20, STATIC)
        rect1.orientation = np.pi - deg2rad(angle)
        rect2 = Rectangle(pos + (0, -length // 3), length, 20, STATIC)
        rect2.orientation = np.pi + deg2rad(angle)
        self.fixed.append(rect1)
        self.fixed.append(rect2)


class ThickStones(EmptyWorld):
    """
    World with thick stones blocking the way
    """

    def __init__(self, obj_maker, sampler, cfg=CFG, seed_env=None):

        super().__init__(obj_maker, sampler, cfg, seed_env)
        self._add_stones()

    def _add_stones(self):
        stones = RandomObstacleGroup(
            np.array([EMPTY + 120, 200]), WIDTH // 6, HEIGHT // 3.5, 3, 3, radius=200)
        stones.color = (100, 100, 100)
        self.fixed.append(stones)


class StandardStones(EmptyWorld):
    """
    World with standard stones blocking the way
    """

    def __init__(self, obj_maker, sampler, cfg=CFG, seed_env=None):

        super().__init__(obj_maker, sampler, cfg, seed_env)
        self._add_stones()

    def _add_stones(self):
        stones = RandomObstacleGroup(
            np.array([EMPTY + 120, 30]), WIDTH // 6, HEIGHT // 3.5, 4, 4, radius=130)
        stones.color = (100, 100, 100)
        self.fixed.append(stones)


class NonConvexWorld(EmptyWorld):
    """
    World with non-convex obstacles
    """

    def __init__(self, obj_maker, sampler, cfg=CFG, seed_env=None):

        super().__init__(obj_maker, sampler, cfg, seed_env)
        self._add_non_convex()

    def _add_non_convex(self):
        self._add_v_shape(np.array([WIDTH // 2, HEIGHT // 2]), 420, 35)
        self._add_v_shape(np.array([WIDTH // 2 - 400, HEIGHT // 3]), 400, 60)
        self._add_v_shape(
            np.array([WIDTH // 2 + 200, HEIGHT // 2 - 300]), 200, 30)
        self._add_v_shape(
            np.array([WIDTH // 2 + 200, HEIGHT // 2 + 300]), 200, 30)
        self._add_v_shape(
            np.array([WIDTH // 2 - 600, HEIGHT // 2 + 300]), 200, 40)

        for i in range(5):
            self._add_v_shape(
                np.array([WIDTH - EMPTY - 200, HEIGHT - i * 200]), 80, 40)

    def _add_v_shape(self, pos, length, angle):
        rect1 = Rectangle(pos + (0, length // 4), length, 20, STATIC)
        rect1.orientation = np.pi - deg2rad(angle)
        rect2 = Rectangle(pos + (0, -length // 4), length, 20, STATIC)
        rect2.orientation = np.pi + deg2rad(angle)
        self.fixed.append(rect1)
        self.fixed.append(rect2)


str2map = {
    "empty": EmptyWorld,
    "non_convex": NonConvexWorld,
    "standard_stones": StandardStones,
    "thick_stones": ThickStones,
    "piped": PipedWorld
}


def get_map(name: str, cfg):
    """
    Get map by name
    :param name: name of the map
    :param cfg: config manager
    :return: map
    """
    return str2map[name](cfg)
    # if name == "empty":
    #     return EmptyWorld(cfg)
    # if name == "non_convex":
    #     return NonConvexWorld(cfg)
    # if name == "standard_stones":
    #     return StandardStones(cfg)
    # if name == "thick_stones":
    #     return ThickStones(cfg)
    # if name == "piped":
    #     return PipedWorld(cfg)
    # raise ValueError(f"Unknown map {name}")


if __name__ == "__main__":
    import pygame

    def maker():
        return Rectangle(START, 20, 20, DYNAMIC)
    sampler = NDIMSampler(np.array([0, 0]), np.array([WIDTH, HEIGHT]))
    ew = StandardStones(maker, sampler)
    mover = PMRectangleController(ew.obj, moving_force=1000)
    # ew = NonConvexWorld(cfg)
    # ew = ThickStones(cfg)
    # ew = PipedWorld(cfg)
    sim = ew.get_sim()
    dbg = DebugViewer(sim, realtime=True)
    dbg.controller = mover

    for i in range(10000):
        # if i % 10 == 0:
        #     print(cable.outer_collision_idxs)
        # print(calc_stretch_index(cable.position, dist_matrix))
        if sim.step():
            break
    # show_sim(ew.sim)
