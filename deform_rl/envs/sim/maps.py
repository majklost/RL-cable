import numpy as np
import pymunk
from .objects import *
from .utils.PM_rectangle_controller import PMRectangleController
from .utils.PM_cable_controller import PMCableController
# from  import ConfigManager
from .utils.seed_manager import init_manager
from .simulator import Simulator
# from deform_plan.utils.PM_space_visu import show_sim
from .utils.PM_debug_viewer import DebugViewer
from .samplers.bezier_sampler import BezierSampler
from .samplers.ndim_sampler import NDIMSampler


# suggested config for these maps
CABLE_LENGTH = 300
WIDTH = 1900
HEIGHT = 800
NUM = 20
UPDATED_CFG = {

    'width': WIDTH,
    'height': HEIGHT,
    'FPS': 60,
    'gravity': 0,
    'damping': .15,
    'collision_slope': 0.01,
    'CABLE_LENGTH': CABLE_LENGTH,

}

# plan - 320 px start, 320 px end; middle is obstacle course
EMPTY = 320
START = (EMPTY - 10, HEIGHT // 2 - CABLE_LENGTH // 2)
END = (WIDTH - EMPTY // 2, HEIGHT // 2)
MARGIN = 50


def deg2rad(deg):
    return deg * np.pi / 180


class EmptyWorld:
    """
    Empty world
    """

    def __init__(self, cfg=UPDATED_CFG, rectangle=False):
        self.cfg = cfg
        self.cfg.update(UPDATED_CFG)
        self._init_manager()
        self.fixed = []
        self.movable = []
        self._add_basic()
        if rectangle:
            print("rectangle")
            self._add_rect()
        else:
            print("cable")
            self._add_cable()

        self._sampler = BezierSampler(self.cable.length, NUM, np.array(
            [MARGIN, MARGIN, 0]), np.array([cfg["width"] - MARGIN, cfg["height"] - MARGIN, 2 * np.pi]))
        self._goal_points = self._sampler.sample(
            x=END[0], y=END[1], angle=0, fixed_shape=True)
        self._in_cnt = 0

    @staticmethod
    def get_movable_idx():
        return 0

    def _init_manager(self):
        seed_env = 90
        init_manager(seed_env, 30)

    def _begin_col(self, arbiter, space, data):
        self._in_cnt += 1
        return False

    def _on_exit(self, arbiter, space, data):
        self._in_cnt -= 1
        return False

    def _add_basic(self):
        boundings = Boundings(self.cfg["width"], self.cfg["height"])
        # end_platform = Rectangle(
        #     np.array([WIDTH - EMPTY // 2 - 10, HEIGHT // 2]), EMPTY, HEIGHT, DYNAMIC, True)
        # end_platform = Rectangle(
        #     np.array([EMPTY // 2 - 10, HEIGHT // 2]), EMPTY, HEIGHT, KINEMATIC, True)
        # end_platform.color = (237, 186, 31)
        # end_platform.set_collision_type(5)

        self.fixed.append(boundings)
        # self.fixed.append(end_platform)

    def _add_rect(self):
        self.cable = Rectangle(START, 20, 20, DYNAMIC)
        self.cable.color = (0, 0, 255)
        # self.cable.set_collision_type(1)
        self.movable.append(self.cable)

    def _add_cable(self):
        self.cable = Cable(START, CABLE_LENGTH,
                           NUM, thickness=5, angle=np.pi)
        # self.cable = Rectangle(START, 20, 20, DYNAMIC)
        self.cable.color = (0, 0, 255)
        self.cable.set_collision_type(1)
        self._start_points = self.cable.position.copy()
        self.movable.append(self.cable)

    @staticmethod
    def get_start():
        return START

    @staticmethod
    def get_goal():
        return END

    def get_start_points(self):
        return self._start_points

    def get_goal_points(self):
        return self._goal_points

    def get_sim(self):
        self.sim = Simulator(self.cfg, self.movable, self.fixed,
                             threaded=False, unstable_sim=True)
        self.sim.add_custom_handler(self._begin_col, self._on_exit, 1, 5)
        return self.sim


class PipedWorld(EmptyWorld):
    """
    World with narrow passages between 'rooms' some feasible, some not
    """

    def __init__(self, cfg):
        cfg.update({
            "seed_env": 50,
        })
        super().__init__(cfg)
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

    def __init__(self):

        super().__init__()
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

    def __init__(self, rectangle=False):
        super().__init__(rectangle=rectangle)
        self._add_stones()

    def _add_stones(self):
        stones = RandomObstacleGroup(
            np.array([EMPTY + 120, 30]), WIDTH // 6, HEIGHT // 3.5, 4, 4, radius=130)
        stones.color = (100, 100, 100)
        self.fixed.append(stones)


class AlmostEmptyWorld(EmptyWorld):
    """
    World with a few obstacles
    """

    def __init__(self):
        super().__init__()
        self._add_obstacles()

    def _init_manager(self):
        # init_manager(10, 20)
        pass

    def _add_obstacles(self):
        stones = RandomObstacleGroup(
            np.array([EMPTY + 120, 60]), WIDTH // 4, HEIGHT // 2.5, 3, 3, radius=100)
        stones.color = (100, 100, 100)
        self.fixed.append(stones)

    def _check_validity(self, pos):
        if not hasattr(self, "sim"):
            raise ValueError("Simulator not initialized, use get_sim() first")

        b = pymunk.Body()
        circ = pymunk.Circle(b, 10)

        for p in pos:
            if not (MARGIN < p[0] < WIDTH - MARGIN and MARGIN < p[1] < HEIGHT - MARGIN):
                return False

            b.position = p.tolist()
            res = self.sim._space.shape_query(circ)
            if res:
                return False
        return True

    def reset_start(self):
        valid = False
        while not valid:
            self._start_points = self._sampler.sample()
            valid = self._check_validity(self._start_points)
        self.cable.position = self._start_points

    def reset_goal(self):
        valid = False
        while not valid:
            self._goal_points = self._sampler.sample()
            valid = self._check_validity(self._goal_points)

    def get_goal_points(self):
        return self._goal_points


class NonConvexWorld(EmptyWorld):
    """
    World with non-convex obstacles
    """

    def __init__(self, cfg):
        cfg.update({
            "seed_env": 51,
        })
        super().__init__(cfg)
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

    # ew = EmptyWorld()
    # ew = StandardStones(cfg)
    # ew = NonConvexWorld(cfg)
    # ew = ThickStones()
    ew = AlmostEmptyWorld()
    sim = ew.get_sim()
    ew.reset_start()
    ew.reset_goal()
    # ew = PipedWorld(cfg)
    mover = PMCableController(ew.cable, moving_force=1000)
    # mover = PMRectangleController(ew.cable, moving_force=1000)
    dbg = DebugViewer(sim, realtime=True)
    dbg.controller = mover

    def draw_gpoints(surf): return [pygame.draw.circle(
        surf, (0, 0, 255), gpoint, 5) for gpoint in ew._goal_points]
    dbg.draw_clb = draw_gpoints
    # show_sim(ew.get_sim(),clb=draw_gpoints)

    for i in range(10000):
        # if i % 10 == 0:
        #     print(cable.outer_collision_idxs)
        # print(calc_stretch_index(cable.position, dist_matrix))
        if sim.step():
            break
        if ew.cable.outer_collision_idxs:
            ew.reset_start()
            ew.reset_goal()

    # show_sim(ew.sim)
