import numpy as np


from ..utils.common_utils import rot_matrix
from .base_sampler import BaseSampler
from .ndim_sampler import NDIMSampler
from ..utils.seed_manager import manager


def create_bezier(A, B, C, D):
    """
    create beziér curve with control points A,B,C,D
    """
    def bezier(t):
        return (1-t)**3*A + 3*(1-t)**2*t*B + 3*(1-t)*t**2*C + t**3*D
    return bezier


class BezierSampler(BaseSampler):
    """
    algorithm to create a path that is non intersecting
    1) [0,0] and [20,0] are endpoints
    2) create rectangle that one side is collinear with the line between the points, same length
    and second side is perpendicular to the line twice the length
    |-|
    A-D
    |-|
    3) sample 2 points B,C on the rectangle sides
    4) form a bezier curve with control points A, B, C, D
    5) bezier curve is parametrize by t in [0,1]
    6) create n+1 points on curve by equidistant sampling of t
    7) MAYBE - TODO - create mapping estimate by https://gamedev.stackexchange.com/questions/5373/moving-ships-between-two-planets-along-a-bezier-missing-some-equations-for-acce/5427#5427
    8) get centers of the n segments between points,these are midpoints of cable segments, get also slopes in these points
    9) scale all segments to fit the desired cable length - this also fixes non uniform lengths of segments
    10) return the points

    """

    def __init__(self, cable_length: int,
                 cable_segments_num: int,
                 lower_bounds: np.array,
                 upper_bounds: np.array,
                 fixed_seed=25):
        super().__init__()
        self.given_seed = manager().get_seed(self.__class__.__name__)
        self.rng = np.random.default_rng(self.given_seed)

        self.cable_length = cable_length
        self.cable_segments_num = cable_segments_num
        self.segment_length = cable_length / cable_segments_num
        self.last_sampled = []
        self.last_angles = []
        self.ndim_sampler = NDIMSampler(lower_bounds, upper_bounds)
        self.fixed_seed = fixed_seed  # for sampling goal points
        # analytics
        self._specific_queries = 0
        self._random_queries = 0

    def sample(self, x=None, y=None, angle=None, fixed_shape=False):
        if x is not None or y is not None or angle is not None:
            return self._sample_goal(x, y, angle, fixed_shape)

        xo, yo, angleo = self.ndim_sampler.sample()
        return self._sample_inner(xo, yo, angleo)

    def _sample_inner(self, x, y, angle):
        self._random_queries += 1
        curve_points = self._get_curve_points(x, y, angle)
        directions = self._calc_directions(curve_points)

        dir_lengths = [np.linalg.norm(d) for d in directions]
        coefs = [self.segment_length / dl for dl in dir_lengths]
        new_dirs = [d * c for d, c in zip(directions, coefs)]
        self.last_angles = self._dirs_to_angles(new_dirs)
        self.last_sampled = self._create_midpoints(
            self._dirs_to_points(curve_points[0], new_dirs))
        mean = np.mean(self.last_sampled, axis=0)-np.array((x, y))
        self.last_sampled -= mean
        return self.last_sampled

    def _sample_goal(self, x, y, angle, fixed_shape=False):
        self._specific_queries += 1
        # print("Sampling goal with fixed seed")
        xf, yf, anglef = self.ndim_sampler.sample()
        if x is not None:
            xf = x
        if y is not None:
            yf = y
        if angle is not None:
            anglef = angle
        if fixed_shape:
            print("Fixed SHAPE")
            old_rng = self.rng
            self.rng = np.random.default_rng(self.fixed_seed)
            points = self._sample_inner(xf, yf, anglef)
            self.rng = old_rng
        else:
            points = self._sample_inner(xf, yf, anglef)
        return points

    def _get_curve_points(self, x, y, angle):
        x_controls = self.rng.uniform(0, 20, 2)
        y_controls = self.rng.uniform(-20, 20, 2)
        A = np.array([0, 0])
        B = np.array([x_controls[0], y_controls[0]])
        C = np.array([x_controls[1], y_controls[1]])
        D = np.array([20, 0])
        bezier = create_bezier(A, B, C, D)

        points = [rot_matrix(angle) @ bezier(t) + np.array([x, y])
                  for t in np.linspace(0, 1, self.cable_segments_num + 1)]
        return points

    @staticmethod
    def _calc_directions(points):
        directions = [points[i + 1] - points[i]
                      for i in range(len(points) - 1)]
        return directions

    @staticmethod
    def _dirs_to_points(start_point, directions):
        points = [start_point]
        for d in directions:
            points.append(points[-1] + d)
        return points

    @staticmethod
    def _create_midpoints(points):
        midpoints = np.array(
            [(points[i] + points[i + 1]) / 2 for i in range(len(points) - 1)])
        return midpoints

    @staticmethod
    def _dirs_to_angles(directions):
        """
        directions to angle with x-axis
        :param directions:
        :return:
        """
        return [np.arctan2(d[1], d[0]) for d in directions]

    def analytics(self):
        return {
            "specific_queries": self._specific_queries,
            "random_queries": self._random_queries
        }
