from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import numpy as np
import os
# from scipy.spatial.transform import Rotation as R
# print(" ".join(R.from_euler('y', 45, degrees=True).as_quat().astype('str')))
model = load_model_from_path("xmls/quad.xml")
sim = MjSim(model)
SIM_TIMESTEP=0.001
SIM_DUARTION_SECS=5

viewer = MjViewer(sim)

sim_state = sim.get_state()

def do_simulation(action, n_frames):
    sim.data.ctrl[:] = action
    for _ in range(n_frames):
        sim.step()
        viewer.render()

def set_state(self, qpos, qvel):
    assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
    old_state = sim.get_state()
    new_state = MjSimState(
        old_state.time, qpos, qvel, old_state.act, old_state.udd_state
    )
    sim.set_state(new_state)
    sim.forward()

quad_id=sim.model.body_name2id('quadrotor')
mass=sim.model.body_mass[quad_id]
print('model mass: ', mass)
num_actuators=4
gear_ratio=2.
nominal_thrust = (mass*9.81)/(num_actuators*gear_ratio)

while True:
    sim.set_state(sim_state)
    action=np.repeat(nominal_thrust, 4)
    do_simulation(action, int(SIM_DUARTION_SECS/SIM_TIMESTEP))