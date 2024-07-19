# imports
import brl_gripper as bg
import mujoco as mj
import numpy as np
import os
import time
import pandas as pd

# controller
from controllers.grab_and_lift.GrabLiftFSM import GrabLiftFSM

# simulation for set of parameters, wrapped in a function
def run_contact_test(sim_params, num_grasps=1, viewer=True, realtime=True):

    # parse sim params, physical params for better storage in data log
    sim_params_f = []
    for param in sim_params:
        param = param.replace("\"", "").replace(" ",",")
        param = param.split(",")
        param = [float(p) for p in param]
        if len(param) == 1:
            param = param[0]
        sim_params_f.append(param)

    # calculate diagonal inertia for object, cube with side length 0.06m
    mass = sim_params_f[6]
    diag_inertia = (1/6)*mass*(0.06**2)
    diaginertias = "\""+str(diag_inertia)+" "+str(diag_inertia)+" "+str(diag_inertia)+"\""

    # separate logging setup from GP
    log_header = ['t_real', 't_sim', 'cycle', 'state']
    log_header += ['obj_px', 'obj_py', 'obj_pz'] #object_pos.tolist() # 3 x 1
    log_header += ['obj_R11', 'obj_R12', 'obj_R13', 'obj_R21', 'obj_R22', 'obj_R23', 'obj_R31', 'obj_R32', 'obj_R33'] #object_R.tolist() # 9 x 1
    log_header += ['obj_vx', 'obj_vy', 'obj_vz', 'obj_wx', 'obj_wy', 'obj_wz'] #object_vel.tolist() # 6 x 1
    log_header += ['obj_Fx', 'obj_Fy', 'obj_Fz', 'obj_Mx', 'obj_My', 'obj_Mz'] #object_force.tolist() # 6 x 1
    log_header += ['base_px', 'base_py', 'base_pz'] #base_pos.tolist() # 3 x 1
    log_header += ['base_vx', 'base_vy', 'base_vz', 'base_wx', 'base_wy', 'base_wz'] #base_vel.tolist() # 6 x 1
    log_header += ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9'] #gripper_pos.tolist() # 9 x 1
    log_header += ['qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6', 'qd7', 'qd8', 'qd9'] #gripper_vel.tolist() # 9 x 1
    log_header += ['tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6', 'tau7', 'tau8', 'tau9'] #gripper_tau.tolist() # 9 x 1
    log_header += ['l_s_Fx', 'l_s_Fy', 'l_s_Fz'] #contact_force_left.tolist() # 3 x 1
    log_header += ['l_s_th', 'l_s_ph'] #contact_angle_left.tolist() # 2 x 1
    log_header += ['l_px', 'l_py', 'l_pz'] #pl_cur.tolist() # 3 x 1
    log_header += ['l_R11', 'l_R12', 'l_R13', 'l_R21', 'l_R22', 'l_R23', 'l_R31', 'l_R32', 'l_R33'] #Rl_cur.tolist() # 9 x 1
    log_header += ['l_vx', 'l_vy', 'l_vz'] #vl_cur.tolist() # 3 x 1
    log_header += ['l_Fx', 'l_Fy', 'l_Fz'] #Fl_cur.tolist() # 3 x 1
    log_header += ['r_s_Fx', 'r_s_Fy', 'r_s_Fz'] #contact_force_right.tolist() # 3 x 1
    log_header += ['r_s_th', 'r_s_ph'] #contact_angle_right.tolist() # 2 x 1
    log_header += ['r_px', 'r_py', 'r_pz'] #pr_cur.tolist() # 3 x 1
    log_header += ['r_R11', 'r_R12', 'r_R13', 'r_R21', 'r_R22', 'r_R23', 'r_R31', 'r_R32', 'r_R33'] #Rr_cur.tolist() # 9 x 1
    log_header += ['r_vx', 'r_vy', 'r_vz'] #vr_cur.tolist() # 3 x 1
    log_header += ['r_Fx', 'r_Fy', 'r_Fz'] #Fr_cur.tolist() # 3 x 1

    log = {}

    # add object to model, set sim_params for this test
    xml_path = os.path.join(bg.assets.ASSETS_DIR, 'scene')
    xml_string = """
    <mujoco model="scene">
        <include file=\"""" + xml_path + ".xml\"" + """/>
        <!-- CUBE -->
        <worldbody>
            <body name="object" pos="0.23 0 0.05">
                <joint type="free" name="object" group="3" stiffness="0" damping="0" frictionloss="0" armature="0"/>
                <inertial pos="0 0 0" mass="""+sim_params[6]+""" diaginertia="""+diaginertias+"""/>
                <geom name="object" type="box" group="3" size="0.03 0.03 0.03" rgba="0.7 0.2 0.1 0.6" contype="1" conaffinity="1" condim="""+sim_params[4]+ \
                    """ priority="2" friction="""+sim_params[7]+""" solimp="""+sim_params[2]+"""  solref="""+sim_params[3]+"""/>
            </body>
        </worldbody>
        <option impratio="""+sim_params[1]+""" timestep="""+sim_params[0]+""" integrator="implicitfast" cone="elliptic" solver="Newton" noslip_iterations="""+sim_params[5]+""">
            <flag contact="enable" override="disable" multiccd="disable"/>
        </option>
    </mujoco>
    """
    mj_model = mj.MjModel.from_xml_string(xml_string)
    # platform
    GP = bg.GripperPlatform(mj_model, viewer_enable=viewer)
    GP.enforce_real_time_sim = realtime
    # controller
    controller = GrabLiftFSM()
    # add variables to GP
    GP.num_cycles = 0
    # start log
    log_lines = []
    log['sim_params'] = sim_params_f
    log['num_grasps'] = num_grasps

    # start experiment
    GP.initialize()
    controller.begin(GP)
    GP.apply_control()
    GP.sync_viewer()
    print("Starting main loop.")
    real_start_time = time.time()
    sim_start_time = GP.time()

    while GP.num_cycles <= num_grasps:         # TODO: no viewer?
        if not GP.paused:
            # step in time to update data from hardware or sim
            GP.step()
            # run controller and update commands
            GP.dt_comp = 0.0 # for real-time simulation
            if GP.run_control:
                control_start_time = GP.time()
                GP.run_control = False
                GP.sync_data()
                controller.update(GP)
                GP.apply_control()
                GP.log_data()

                # custom logging here
                t_log_sim = GP.time() - sim_start_time
                t_log_real = time.time() - real_start_time

                object_pos = GP.mj_data.body('object').xpos
                object_R = GP.mj_data.body('object').xmat
                object_vel = GP.mj_data.joint('object').qvel

                base_pos = GP.gr_data.kinematics['base']['p']
                base_R = GP.gr_data.kinematics['base']['R'].flatten()
                base_vel = GP.mj_data.joint('floating_2').qvel

                gripper_pos = GP.gr_data.get_q(GP.gr_data.all_idxs)
                gripper_vel = GP.gr_data.get_qd(GP.gr_data.all_idxs)
                gripper_tau = GP.gr_data.get_tau(GP.gr_data.all_idxs)

                # TODO: have sensor store force in world frame, rotation matrix for contact frame?
                contact_force_left = GP.gr_data.sensors['l_dip'].contact_force
                contact_force_right = GP.gr_data.sensors['r_dip'].contact_force
                contact_angle_left = GP.gr_data.sensors['l_dip'].contact_angle
                contact_angle_right = GP.gr_data.sensors['r_dip'].contact_angle

                # TODO: confirm this?
                object_force = GP.mj_data.joint('object').qfrc_constraint # + GP.mj_data.joint('object').qfrc_smooth

                qdl_cur = GP.gr_data.get_qd(GP.gr_data.l_idxs)
                taul_cur = GP.gr_data.get_tau(GP.gr_data.l_idxs)
                pl_cur = GP.gr_data.kinematics['l_dip_tip']['p']
                Rl_cur = GP.gr_data.kinematics['l_dip_tip']['R'].flatten()
                Jl_cur = GP.gr_data.kinematics['l_dip_tip']['Jacp']
                vl_cur = Jl_cur @ qdl_cur
                Fl_cur = np.linalg.pinv(Jl_cur.T) @ taul_cur
                qdr_cur = GP.gr_data.get_qd(GP.gr_data.r_idxs)
                taur_cur = GP.gr_data.get_tau(GP.gr_data.r_idxs)
                pr_cur = GP.gr_data.kinematics['r_dip_tip']['p']
                Rr_cur = GP.gr_data.kinematics['r_dip_tip']['R'].flatten()
                Jr_cur = GP.gr_data.kinematics['r_dip_tip']['Jacp']
                vr_cur = Jr_cur @ qdr_cur
                Fr_cur = np.linalg.pinv(Jr_cur.T) @ taur_cur

                # TODO: what else to log?

                # convert all data to lists for logging
                log_line = [t_log_real, t_log_sim, GP.num_cycles, controller.current_state_idx]

                log_line += object_pos.tolist() # 3 x 1
                log_line += object_R.tolist() # 9 x 1
                log_line += object_vel.tolist() # 6 x 1
                log_line += object_force.tolist() # 6 x 1

                log_line += base_pos.tolist() # 3 x 1
                log_line += base_vel.tolist() # 6 x 1

                log_line += gripper_pos.tolist() # 9 x 1
                log_line += gripper_vel.tolist() # 9 x 1
                log_line += gripper_tau.tolist() # 9 x 1

                log_line += contact_force_left.tolist() # 3 x 1
                log_line += contact_angle_left.tolist() # 2 x 1
                log_line += pl_cur.tolist() # 3 x 1
                log_line += Rl_cur.tolist() # 9 x 1
                log_line += vl_cur.tolist() # 3 x 1
                log_line += Fl_cur.tolist() # 3 x 1

                log_line += contact_force_right.tolist() # 3 x 1
                log_line += contact_angle_right.tolist() # 2 x 1
                log_line += pr_cur.tolist() # 3 x 1
                log_line += Rr_cur.tolist() # 9 x 1
                log_line += vr_cur.tolist() # 3 x 1
                log_line += Fr_cur.tolist() # 3 x 1

                log_lines.append(log_line)

                GP.dt_comp += GP.time() - control_start_time
            # sync viewer
            if GP.run_viewer_sync:
                viewer_sync_start_time = GP.time()
                GP.run_viewer_sync = False
                GP.sync_viewer()
                GP.dt_comp += GP.time() - viewer_sync_start_time

    print("t_sim:", GP.time()-sim_start_time, ", t_real:", time.time()-real_start_time)

    # store data
    log['data'] = pd.DataFrame(log_lines, columns=log_header)

    # end experiment
    GP.shutdown()

    # clean up, delete big things, except for log
    del(mj_model)
    del(GP)
    del(controller)

    # return log of data for all idxs
    return log
