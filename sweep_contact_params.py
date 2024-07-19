import brl_gripper as bg
import mujoco as mj
import numpy as np
import os
import pickle
import time
import csv
from datetime import datetime as dt
import itertools
import pandas as pd

# actual sim run function
from contact_test import run_contact_test

sweep_start = dt.now()

# number of grasps per set of parameters
num_grasps = 3

# set up lists of parameters to iterate over
timesteps = ["\"0.0001\"",
             "\"0.00025\"",
            "\"0.0005\"",
            "\"0.001\""]
impratios = ["\"2\"",
            "\"3\"",
            "\"4\"",
            "\"5\""]
solimps = ["\"0.95 0.99 0.001 0.5 2\"",
            "\"0.98 0.999 0.001 0.5 2\""] # (d0​, dwidth​, width, midpoint, power)
solrefs = ["\"0.002 1\"",
            "\"0.003 1\"",
            "\"0.004 1\"",
            "\"0.005 1\"",
            "\"0.006 1\"",
            "\"0.007 1\"",
            "\"0.008 1\""] # (timeconst,dampratio)
condims = ["\"4\""]
noslip_iterations = ["\"0\""]
masses = ["\"0.5\"",
            "\"0.70\"",
            "\"0.85\"",
            "\"1.0\""] # will calculate diaginertias from this
frictions = ["\"1.0 0.01 0.0001\"",
             "\"1.5 0.01 0.0001\"",
             "\"1.0 0.02 0.0001\"",
             "\"1.5 0.02 0.0001\"",
             "\"1.0 0.05 0.0001\"",
             "\"1.5 0.05 0.0001\"",] # (sliding, torsion, rolling)
# - friction[0]: [0.5, 0.75, 1]
# - friction[1]: [0.01, 0.015, 0.02]
# - friction[2]: [0.0001, 0.001, 0.01]

# NOTE:
# from solimp and solref, constraint damping and stiffness:
# b = 2/(dwidth * timeconst)
# k = d(r)/(dwidth^2 * timeconst*2 * dampratio^2), d(r) = f(d0, dwidth, width, midpoint, power)

num_combos = len(timesteps) * len(impratios) * len(solimps) * len(solrefs) * len(condims) * len(noslip_iterations) * len(masses) * len(frictions)
num_tests = num_combos * num_grasps
print("Total number of combinations:", num_tests)


# NOTE: current params: timestep=0.001, impratio=10, solimp=0.95 0.99 0.001 0.5 2, solref=0.005 1, condim=4, noslip_iterations=0, mass=0.1, friction=1 0.02 0.0001
# NOTE: sweep can run at >3.5x real time, with or without viewer
# NOTE: data log for one test is about 22MB
# NOTE: full sweep took 2hrs 30min on lab CPU, logs totalled 16GB

# set up summary log
# general sim data
sum_header = ['num_grasps', 'timestep', 'impratio', \
                'solimp_dmin', 'solimp_dmax', 'solimp_width', 'solimp_midpoint', 'solimp_power', \
                'solref_timeconst', 'solref_dampratio', \
                'condim', 'noslip_iterations', \
                'mass', 'friction_sliding', 'friction_torsion', 'friction_rolling']
sum_header += ['speedup']

# grasping
sum_header += ['grasp_l_Ft_f', 'grasp_l_Fn_f', 'grasp_r_Ft_f', 'grasp_r_Fn_f']
sum_header += ['grasp_obj_Fx_f', 'grasp_obj_Fy_f', 'grasp_obj_Fz_f']
sum_header += ['grasp_mean_pen']

# lifting
sum_header += ['lift_mean_rel_obj_v', 'lift_var_rel_obj_v']
sum_header += ['lift_mean_obj_w', 'lift_var_obj_w']
sum_header += ['lift_mean_l_Ft', 'lift_mean_l_Fn', 'lift_mean_r_Ft', 'lift_mean_r_Fn']
sum_header += ['lift_var_l_Ft', 'lift_var_l_Fn', 'lift_var_r_Ft', 'lift_var_r_Fn']
sum_header += ['lift_mean_obj_Fx', 'lift_mean_obj_Fy', 'lift_mean_obj_Fz']
sum_header += ['lift_var_obj_Fx', 'lift_var_obj_Fy', 'lift_var_obj_Fz']
sum_header += ['lift_success']

# holding
sum_header += ['hold_mean_rel_obj_v', 'hold_mean_obj_w']
sum_header += ['hold_obj_pz_err_f']
sum_header += ['hold_mean_l_Ft', 'hold_mean_l_Fn', 'hold_mean_r_Ft', 'hold_mean_r_Fn']
sum_header += ['hold_l_Ft_f', 'hold_l_Fn_f', 'hold_r_Ft_f', 'hold_r_Fn_f']
sum_header += ['hold_success']

# releasing
sum_header += ['release_max_lin_vel', 'release_max_rot_vel']
sum_header += ['release_obj_max_pen']

sum_log = []

# start iteration
i = 0
for sim_params in itertools.product(timesteps, impratios, solimps, solrefs, condims, noslip_iterations, masses, frictions):
    # # early termination for debugging
    # if i>=10:
    #     break

    for g in range(num_grasps):
        # increment iteration counter
        i+=1
        print("Test "+str(i)+" of "+str(num_tests)+", Params: "+str(sim_params)+", Grasp attempt: "+str(g+1))

        # run simulation with these parameters
        full_log = run_contact_test(sim_params, viewer=False, realtime=False)
        data_log = full_log['data']

        # # TODO: uncomment this to save every test
        # # save test data to a pickle file
        # log_name = "test_data_log_"+str(i)
        # with open(log_dir+log_name, 'wb') as log_file:
        #     pickle.dump(full_log, log_file)

        # collecting data for entire sweep:
        sum_log_line = []

        # general sim data:
        ps = full_log['sim_params']
        sum_log_line += [num_grasps, ps[0], ps[1], \
                            ps[2][0], ps[2][1], ps[2][2], ps[2][3], ps[2][4], \
                            ps[3][0], ps[3][1], \
                            ps[4], ps[5], \
                            ps[6], ps[7][0], ps[7][1], ps[7][2]]
        speedup = data_log['t_sim'].iloc[-1] / data_log['t_real'].iloc[-1]
        sum_log_line += [speedup]

        # grasping:
        grasp_data = data_log[data_log['state']==2]
        # terminal contact forces during grasping, tangential and normal
        l_Ft_f = np.linalg.norm(grasp_data.loc[:, ['l_s_Fx', 'l_s_Fy']].iloc[-1:].values.flatten())
        l_Fn_f = np.abs(grasp_data.loc[:, 'l_s_Fz'].iloc[-1])
        r_Ft_f = np.linalg.norm(grasp_data.loc[:, ['r_s_Fx', 'r_s_Fy']].iloc[-1:].values.flatten())
        f_Fn_f = np.abs(grasp_data.loc[:, 'r_s_Fz'].iloc[-1])
        tip_F_f = [l_Ft_f, l_Fn_f, r_Ft_f, f_Fn_f]
        # terminal object force during grasping
        obj_F_f = grasp_data.loc[:, ['obj_Fx', 'obj_Fy', 'obj_Fz']].iloc[-1:].values.flatten().tolist() # TODO: include moments?
        # min dist between fingertips during grasping
        dist_fingertips = np.linalg.norm( grasp_data.loc[:, ['l_px','l_py','l_pz']].values - grasp_data.loc[:, ['r_px','r_py','r_pz']].values, axis=1)
        mean_fingertip_penetration = [0.5*(np.min(dist_fingertips)-0.06-0.02)] # subtracting object width and fingertip radii

        sum_log_line += tip_F_f + obj_F_f + mean_fingertip_penetration

        # lifting:
        lift_data = data_log[data_log['state']==3]
        # mean of relative lin vels during lifting
        # variance of relative lin vels during lifting
        rel_obj_vel = lift_data.loc[:, ['obj_vx', 'obj_vy', 'obj_vz']].values - lift_data.loc[:, ['base_vx', 'base_vy', 'base_vz']].values
        rel_obj_vel_mag = np.linalg.norm(rel_obj_vel, axis=1)
        rel_obj_vel_mag_stats = [np.mean(rel_obj_vel_mag), np.var(rel_obj_vel_mag)]
        # mean and variance of rot vels during lifting?
        obj_w = lift_data.loc[:, ['obj_wx', 'obj_wy', 'obj_wz']].values
        obj_w_mag = np.linalg.norm(obj_w, axis=1)
        obj_w_mag_stats = [np.mean(obj_w_mag), np.var(obj_w_mag)]
        # mean contact forces (tangential and normal) during lifting
        l_Ft = np.linalg.norm(lift_data.loc[:, ['l_s_Fx', 'l_s_Fy']].values, axis=1)
        l_Fn = np.abs(lift_data.loc[:, 'l_s_Fz'].values)
        r_Ft = np.linalg.norm(lift_data.loc[:, ['r_s_Fx', 'r_s_Fy']].values, axis=1)
        r_Fn = np.abs(lift_data.loc[:, 'r_s_Fz'].values)
        mean_tip_F = [np.mean(l_Ft), np.mean(l_Fn), np.mean(r_Ft), np.mean(r_Fn)]
        var_tip_F = [np.var(l_Ft), np.var(l_Fn), np.var(r_Ft), np.var(r_Fn)]
        # average object force during lifting
        mean_obj_F = np.mean(lift_data.loc[:, ['obj_Fx', 'obj_Fy', 'obj_Fz']].values, axis=0).tolist()
        var_obj_F =np.var(lift_data.loc[:, ['obj_Fx', 'obj_Fy', 'obj_Fz']].values, axis=0).tolist()
        mean_obj_F_mag = [np.linalg.norm(mean_obj_F)]
        # terminal contact forces during holding, tangential and normal
        l_Fn_f = np.abs(lift_data.loc[:, 'l_s_Fz'].iloc[-1])
        f_Fn_f = np.abs(lift_data.loc[:, 'r_s_Fz'].iloc[-1])
        # lift sucess flag
        Fn_thresh = 0.5 # in N
        lift_success = [0.0]
        if (l_Fn_f > Fn_thresh) and (f_Fn_f > Fn_thresh): lift_success = [1.0]

        sum_log_line += rel_obj_vel_mag_stats + obj_w_mag_stats + mean_tip_F + var_tip_F + mean_obj_F + var_obj_F + lift_success

        # holding:
        hold_data = data_log[data_log['state']==4]
        # average object force during holding
        mean_obj_F = np.mean(hold_data.loc[:, ['obj_Fx', 'obj_Fy', 'obj_Fz']].values, axis=0).tolist()
        mean_obj_F_mag = [np.linalg.norm(mean_obj_F)]
        # average relative box vel during holding?
        mean_rel_obj_lin_vel = np.mean(hold_data.loc[:, ['obj_vx', 'obj_vy', 'obj_vz']].values-hold_data.loc[:, ['base_vx', 'base_vy', 'base_vz']].values, axis=0)
        mean_rel_obj_lin_vel_mag = [np.linalg.norm(mean_rel_obj_lin_vel)]
        mean_obj_w = np.mean(hold_data.loc[:, ['obj_wx', 'obj_wy', 'obj_wz']].values, axis=0)
        mean_obj_w_mag = [np.linalg.norm(mean_obj_w)]
        # terminal box height during holding
        obj_p_f = hold_data.loc[:, ['obj_pz', 'obj_py','obj_pz']].iloc[-1:].values.flatten().tolist()
        obj_pz_err_f = [np.abs(obj_p_f[2]-0.25)] # subtracting desired object height
        # mean contact forces during holding
        l_Ft = np.linalg.norm(hold_data.loc[:, ['l_s_Fx', 'l_s_Fy']].values, axis=1)
        l_Fn = np.abs(hold_data.loc[:, 'l_s_Fz'].values)
        r_Ft = np.linalg.norm(hold_data.loc[:, ['r_s_Fx', 'r_s_Fy']].values, axis=1)
        r_Fn = np.abs(hold_data.loc[:, 'r_s_Fz'].values)
        mean_tip_F = [np.mean(l_Ft), np.mean(l_Fn), np.mean(r_Ft), np.mean(r_Fn)]
        # terminal contact forces during holding, tangential and normal
        l_Ft_f = np.linalg.norm(hold_data.loc[:, ['l_s_Fx', 'l_s_Fy']].iloc[-1:].values.flatten())
        l_Fn_f = np.abs(hold_data.loc[:, 'l_s_Fz'].iloc[-1])
        r_Ft_f = np.linalg.norm(hold_data.loc[:, ['r_s_Fx', 'r_s_Fy']].iloc[-1:].values.flatten())
        f_Fn_f = np.abs(hold_data.loc[:, 'r_s_Fz'].iloc[-1])
        tip_F_f = [l_Ft_f, l_Fn_f, r_Ft_f, f_Fn_f]
        # hold sucess flag
        Fn_thresh = 0.5 # in N
        hold_success = [0.0]
        if (l_Fn_f > Fn_thresh) and (f_Fn_f > Fn_thresh): hold_success = [1.0]

        sum_log_line += mean_rel_obj_lin_vel_mag + mean_obj_w_mag + obj_pz_err_f + mean_tip_F + tip_F_f + hold_success

        # releasing:
        release_data = data_log[data_log['state']==5]
        # maximum magnitude of lin and rot box vel during release # TODO: save max z-vel? from after landing and bouncing?
        lin_vel_mag = np.linalg.norm(release_data.loc[:, ['obj_vx', 'obj_vy', 'obj_vz']].values, axis=1)
        rot_vel_mag = np.linalg.norm(release_data.loc[:, ['obj_wx', 'obj_wy', 'obj_wz']].values, axis=1)
        max_vel_mags = [np.max(lin_vel_mag), np.max(rot_vel_mag)]
        # minimum box height during release # TODO: consider box orientation here?
        obj_min_height = np.min(release_data.loc[:, 'obj_pz'].values) - 0.03 # subtracting cube side length / 2
        obj_max_penetration = [abs(min(0, obj_min_height))] # only consider penetration (negative values)

        sum_log_line += max_vel_mags + obj_max_penetration

        # add log line to summary log
        sum_log.append(sum_log_line)

# create new log directory for each sweep
log_dir = "logs/sweep_"+str(dt.now()).replace(" ", "_")+"/"
os.mkdir(log_dir)
# save summary log as dataframe
sum_data = pd.DataFrame(sum_log, columns=sum_header)
log_name = "summary_log"
with open(log_dir+log_name, 'wb') as log_file:
    pickle.dump(sum_data, log_file)
print("Summary dataframe shape:", sum_data.shape)

# end
sweep_end = dt.now()
sweep_duration = sweep_end - sweep_start
print("Sweep duration:", sweep_duration, "Number of tests ran:", i)
