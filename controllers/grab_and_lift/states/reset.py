# imports
import numpy as np
import mujoco as mj
from controllers.base.baseState import *
from .fsm_config import *

# define state
class Reset(BaseState):
    def __init__(self):
        self.name = "Reset"
        self.enabled = 0

    def enter(self, GP):
        print(self.name)
        self.enabled = 1

    def exit(self, GP):
        self.enabled = 0

    def execute(self, GP):

        # increment cycle counter here
        GP.num_cycles += 1
        print("Controller cycle:", GP.num_cycles)

        # TODO: update mj_model settings based on number of cycles here?

        # directly set finger positions
        GP.mj_data.qpos[7:16] = fsm_params.q_des_default
        GP.mj_data.qvel[7:16] = np.zeros((9,))

        # directly set wrist position
        GP.mj_data.mocap_pos = fsm_params.base_pos_default
        base_quat_default = np.zeros((4,))
        mj.mju_mat2Quat(base_quat_default, fsm_params.base_R_default.flatten())
        GP.mj_data.mocap_quat = base_quat_default

        # directly set block position
        GP.mj_data.qpos[16:19] = fsm_params.obj_pos_default
        obj_quat_default = np.zeros((4,))
        mj.mju_mat2Quat(obj_quat_default, fsm_params.obj_R_default.flatten())
        GP.mj_data.qpos[19:23] = obj_quat_default

        # go to waiting state after reset
        next_state = "Waiting"

        return next_state