import pymunk

from deform_rl.sim.Rectangle_env.simulator import Simulator
from deform_rl.sim.Rectangle_env.objects.rectangle import Rectangle
from deform_rl.sim.Rectangle_env.utils.PM_debug_viewer import DebugViewer
from deform_rl.sim.Rectangle_env.utils.PM_rectangle_controller import PMRectangleController


rect = Rectangle([100, 100], 200, 20, pymunk.Body.DYNAMIC)
sim_cfg = {
    'width': 800,
    'height': 600,
    'FPS': 60,
    'gravity': 0,
    'damping': .15,
    'collision_slope': 0.01,
}
sim = Simulator(sim_cfg, [rect], [], unstable_sim=True)

controller = PMRectangleController(rect, moving_force=20000)

viewer = DebugViewer(sim, realtime=True)
viewer.controller = controller

for i in range(10000):
    print(rect.position)
    if sim.step():
        break
