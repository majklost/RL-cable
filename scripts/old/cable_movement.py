import numpy as np

from deform_rl.envs.sim.simulator import Simulator
from deform_rl.envs.sim.objects.rectangle import Rectangle
from deform_rl.envs.sim.objects.cable import Cable
from deform_rl.envs.sim.utils.PM_debug_viewer import DebugViewer
from deform_rl.envs.sim.utils.PM_cable_controller import PMCableController
from deform_rl.envs.sim.utils.standard_cfg import sim_cfg
from deform_rl.envs.sim.samplers.bezier_sampler import BezierSampler

# cable = Cable([400, 400], 200, 4, thickness=5, angle=0)
cable = Cable([400, 400], 200, 10, thickness=3,
              angle=np.pi / 4, max_angle=np.pi / 4)
bs = BezierSampler(300, 30, lower_bounds=[
                   0, 0, 0], upper_bounds=[800, 800, 2 * np.pi])
print(bs.sample())

sim = Simulator(sim_cfg, [cable], [], unstable_sim=True)
cable_controller = PMCableController(cable, moving_force=500)


dbg = DebugViewer(sim, realtime=True, render_constraints=False)
dbg.controller = cable_controller

print(cable.position.shape)
print("-" * 10)
flat = cable.position.flatten()
resh = flat.reshape((flat.shape[0] // 2, 2))
# print(np.all(resh == cable.position))

for i in range(10000):
    if sim.step():
        break
    # print(cable.position.shape)
