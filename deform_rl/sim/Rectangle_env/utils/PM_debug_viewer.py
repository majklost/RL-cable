import pymunk
import pygame
from pymunk.pygame_util import DrawOptions

from ..simulator import Simulator
from .PM_rectangle_controller import PMRectangleController


class DebugViewer:
    def __init__(self, simulator: Simulator,
                 render_constraints=False,
                 realtime=False):
        self.simulator = simulator
        self.simulator.debuggerclb = lambda x: self.update_cur(x)
        w, h, self.FPS = self.simulator.get_debug_data()
        self.display = pygame.display.set_mode((w, h))
        self.cur_scene = pygame.surface.Surface((w, h))
        self.draw_options = DrawOptions(self.cur_scene)
        self.realtime = realtime
        self.clock = pygame.time.Clock()
        self.want_running = True
        self.controller = None  # type: PMCableController
        self.drawings = []
        self.draw_clb = None
        if not render_constraints:
            self.draw_options.flags = DrawOptions.DRAW_SHAPES

    def update_cur(self, space: pymunk.Space):
        # new rendered image, will be prepared to be rendered
        self.cur_scene.fill((255, 255, 255))
        space.debug_draw(self.draw_options)
        for d in self.drawings:
            d.draw(self.cur_scene)
        if self.draw_clb:
            self.draw_clb(self.cur_scene)
        self.display.blit(self.cur_scene, (0, 0))
        pygame.display.update()
        if self.realtime:
            self.clock.tick(self.FPS)
        if self.controller is not None:
            self.controller.update()
        else:
            self._mark_end_custom()
        if self.want_end:
            return False
        return True

    def want_end(self):
        return not self.want_running

    def _mark_end_custom(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.want_running = False

    def draw_line(self, start, end, color=(0, 0, 0)):
        self.drawings.append(self._Line(start, end, color))

    def draw_circle(self, pos, radius, color=(0, 0, 0)):
        self.drawings.append(self._Circle(pos, radius, color))

    class _Circle:
        def __init__(self, pos, radius, color):
            self.pos = pos
            self.radius = radius
            self.color = color

        def draw(self, display):
            pygame.draw.circle(display, self.color, self.pos, self.radius)

    class _Line:
        def __init__(self, start, end, color):
            self.start = start
            self.end = end
            self.color = color

        def draw(self, display):
            pygame.draw.line(display, self.color, self.start, self.end)
