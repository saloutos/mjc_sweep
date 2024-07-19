# define config variables for all states here

# imports
import numpy as np

# general params class to add attributes to for each state
class FSMparams:
    pass

fsm_params = FSMparams()

# NOTE: (from gripper data)
# w_idxs = [0] # joint idx for wrist
# l_idxs = [1,2,3,4] # joint idxs for left finger
# r_idxs = [5,6,7,8] # joint idxs for right finger

# default gains for fingers and wrist
fsm_params.kp_default = np.array([3.0, 8.0, 2.5, 2.5, 2.5, 8.0, 2.5, 2.5, 2.5])
fsm_params.kd_default = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

# initial state of fingers, wrist, object
fsm_params.q_des_default = np.array([0.0, 0.0, 0.4, -0.4, -0.0, 0.0, -0.4, 0.4, 0.0])

fsm_params.base_pos_default = np.array([0.0, 0.0, 0.05])
fsm_params.base_R_default = np.eye(3)

fsm_params.obj_pos_default = np.array([0.22, 0.0, 0.031])
fsm_params.obj_R_default = np.eye(3)

# finger pose for grasping, lifting, holding
fsm_params.q_des_grasp = np.array([0.0, 0.0, 0.2, -0.8, -0.8, 0.0, -0.2, 0.8, 0.8])

# finger gains for grasping, lifting, holding (if not defaults?)
fsm_params.kp_grasp = np.array([3.0, 8.0, 2.5, 2.5, 2.5, 8.0, 2.5, 2.5, 2.5])
fsm_params.kd_grasp = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

# wrist pose for holding
fsm_params.base_pos_hold = np.array([0.0, 0.0, 0.25])
fsm_params.base_R_hold = np.eye(3)

# dict of trajectory times for each state
fsm_params.times = {'wait':     0.5,
                    'grasp':    0.5,
                    'lift':     1.0,
                    'hold':     2.0,
                    'release':  1.0} # 5 seconds total