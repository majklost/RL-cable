import pymunk
import pygame
from deform_rl.envs.sim.maps import StandardStones
from deform_rl.envs.sim.utils.PM_rectangle_controller import PMRectangleController
from deform_rl.envs.sim.utils.PM_cable_controller import PMCableController
from deform_rl.envs.sim.utils.PM_debug_viewer import DebugViewer

ew = StandardStones(rectangle=False)
sim = ew.get_sim()
for b in ew.cable.bodies:
    for s in b.shapes:
        s.filter = pymunk.ShapeFilter(categories=0b1)
# mover = PMRectangleController(ew.cable, moving_force=1000)
mover = PMCableController(ew.cable, moving_force=1000)
dbg = DebugViewer(sim, realtime=True)
dbg.controller = mover


def draw_clb(surf):


    # circ = pymunk.shapes.Circle(dummy, 100)

    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_a]:
    #     res = sim._space.shape_query(circ)
    #     print(res)
    # pygame.draw.circle(surf, (0, 255, 0),
    #                    (int(pos[0]), int(pos[1])), 5)
    my_filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
    responses = [sim._space.point_query_nearest(
        x.tolist(), 300, my_filter) for x in ew.cable.position]
    for res in responses:
        if res:
            pygame.draw.circle(surf, (0, 255, 0),
                               (int(res.point.x), int(res.point.y)), 5)


dbg.draw_clb = draw_clb

for i in range(10000):

    if sim.step():
        break
