import pymunk
import numpy as np
import pygame
from deform_rl.envs.sim.maps import StandardStones, EmptyWorld
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
    # responses = [sim._space.point_query_nearest(
    #     x.tolist(), 300, my_filter) for x in ew.cable.position]
    unit_vecs1 = np.array([(np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2))
                          for ang in ew.cable.orientation])
    unit_vecs2 = -1 * unit_vecs1

    orient_vecs = np.array([(np.cos(ang), np.sin(ang))
                            for ang in ew.cable.orientation])

    # unit_vecs = np.concatenate((unit_vecs1, unit_vecs2), axis=0)

    responses1 = [sim._space.segment_query_first(
        pos.tolist(), (pos + 100 * vec).tolist(), 1, my_filter) for pos, vec in zip(ew.cable.position, unit_vecs1)]
    responses2 = [sim._space.segment_query_first(
        pos.tolist(), (pos + 100 * vec).tolist(), 1, my_filter) for pos, vec in zip(ew.cable.position, unit_vecs2)]

    for i in range(len(ew.cable.position)):
        pos = ew.cable.position[i].tolist()
        # pygame.draw.line(surf, (255, 0, 0), pos,
        #                  orient_vecs[i] * 100 + pos, 1)

    for i in range(len(ew.cable.position)):
        pos = ew.cable.position[i].tolist()
        # pygame.draw.line(surf, (255, 0, 0), pos,
        #                  responses1[i].point, 1)
        # pygame.draw.line(surf, (255, 0, 0), pos,
        #                  responses2[i].point, 1)
        if responses1[i]:
            pygame.draw.circle(surf, (0, 255, 0),
                               (int(responses1[i].point.x), int(responses1[i].point.y)), 5)
            pygame.draw.line(surf, (255, 0, 0), pos,
                             responses1[i].point, 1)
        if responses2[i]:
            pygame.draw.circle(surf, (0, 255, 0),
                               (int(responses2[i].point.x), int(responses2[i].point.y)), 5)
            pygame.draw.line(surf, (255, 0, 0), pos,
                             responses2[i].point, 1)

    distances1 = np.array([np.linalg.norm(np.array(
        res.point) - pos) if res else 100 for pos, res in zip(ew.cable.position, responses1)])

    distances2 = np.array([np.linalg.norm(np.array(
        res.point) - pos) if res else 100 for pos, res in zip(ew.cable.position, responses2)])

    if distances1[0] != 100:
        print("side1: ", distances1[0])
    if distances2[0] != 100:
        print("side2: ", distances2[0])
    # for res in responses:
    #     if res:
    #         pygame.draw.circle(surf, (0, 255, 0),
    #                            (int(res.point.x), int(res.point.y)), 5)


dbg.draw_clb = draw_clb
for i in range(10000):

    if sim.step():
        break
