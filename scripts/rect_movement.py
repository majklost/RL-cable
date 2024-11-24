import pymunk

from deform_rl.envs.sim.simulator import Simulator
from deform_rl.envs.sim.objects.rectangle import Rectangle
from deform_rl.envs.sim.utils.PM_debug_viewer import DebugViewer
from deform_rl.envs.sim.utils.PM_rectangle_controller import PMRectangleController
from deform_rl.envs.sim.utils.standard_cfg import sim_cfg




rect = Rectangle([100, 100], 200, 20, pymunk.Body.DYNAMIC)

sim = Simulator(sim_cfg, [rect], [], unstable_sim=True)

controller = PMRectangleController(rect, moving_force=20000)

viewer = DebugViewer(sim, realtime=True)
viewer.controller = controller

for i in range(10000):
    print(rect.position)
    if sim.step():
        break
