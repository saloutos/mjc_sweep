# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

from plot_helpers import *

# load sweep summary file
filename = 'sweep_2024-07-19_07:25:38.928763_summary_log' # most recent sweep

with open(filename, 'rb') as f:
    data = pickle.load(f)

# check it out
# data.columns

# %%
# general plots
plot_one_x_one_y_many_splits(data, 'timestep', 'speedup', ['mass', 'solref_timeconst'])
plot_one_y_for_many_x(data, 'speedup', ['mass', 'impratio', 'solimp_dmin', 'solref_timeconst'], 'timestep')

# %%
# grasping state plots
plot_one_x_one_y_many_splits(data, 'mass', 'grasp_mean_pen', ['solimp_dmin', 'solref_timeconst'])
plot_one_x_one_y_many_splits(data, 'solref_timeconst', 'grasp_mean_pen', ['mass', 'solimp_dmin'])

# %%
# lifting state plots
heavy_data = data.loc[data.mass==1.0]
cat_names = ['lift_mean_rel_obj_v', 'lift_var_rel_obj_v', 'lift_mean_obj_w', 'lift_var_obj_w', 'lift_mean_l_Ft', 'lift_mean_l_Fn', 'lift_mean_obj_Fx', 'lift_mean_obj_Fy', 'lift_mean_obj_Fz']
# plot_one_x_for_many_y(heavy_data, 'solref_timeconst', cat_names, 'timestep')
plot_one_x_for_many_y(data, 'solref_timeconst', cat_names, 'mass')

plot_one_x_one_y_many_splits(data, 'solref_timeconst', 'lift_mean_rel_obj_v', ['mass', 'friction_sliding', 'friction_torsion'])
plot_one_x_one_y_many_splits(data, 'solref_timeconst', 'lift_mean_obj_w', ['mass', 'friction_sliding', 'friction_torsion'])


# %%
# holding state plots
heavy_data = data.loc[data.mass==1.0]

plot_one_x_one_y_many_splits(data, 'solref_timeconst', 'hold_mean_rel_obj_v', ['friction_sliding', 'friction_torsion'])
plot_one_x_one_y_many_splits(data, 'solref_timeconst', 'hold_mean_obj_w', ['friction_sliding', 'friction_torsion'])

plot_one_x_one_y_many_splits(data, 'solref_timeconst', 'hold_obj_pz_err_f', ['mass', 'impratio', 'solimp_dmin', 'friction_sliding', 'friction_torsion'])
plot_one_x_one_y_many_splits(data, 'solref_timeconst', 'hold_mean_l_Fn', ['mass', 'impratio', 'solimp_dmin'])
plot_one_x_one_y_many_splits(data, 'hold_mean_l_Ft', 'hold_mean_l_Fn', ['mass', 'friction_sliding', 'friction_torsion'])

# %%
# release state plots
hold_success_data = data.loc[data.hold_success==1]
plot_one_x_one_y_many_splits(hold_success_data, 'solref_timeconst', 'release_obj_max_pen', ['mass', 'timestep', 'impratio','solimp_dmin', 'friction_sliding', 'friction_torsion'])
# TODO: improve lift success flag

# %%
# summary plot?
# final_data = data.loc[(data.timestep==0.001) & \
#                       (data.impratio==5) & \
#                       (data.solimp_dmin==0.95) & \
#                       (data.solref_timeconst==0.005) & \
#                       (data.condim==4) & \
#                       (data.noslip_iterations==0) ]

# cat_names = ['speedup', 'grasp_mean_pen', \
#             'lift_mean_rel_obj_v', 'lift_var_rel_obj_v', 'lift_mean_obj_w', 'lift_var_obj_w', 'lift_mean_l_Ft', 'lift_mean_l_Fn', 'lift_mean_obj_Fx', 'lift_mean_obj_Fy', 'lift_mean_obj_Fz', \
#             'hold_mean_rel_obj_v', 'hold_mean_obj_w', 'hold_obj_pz_err_f', 'hold_mean_l_Fn', \
#             'release_obj_max_pen', 'release_max_rot_vel']

# plot_one_x_for_many_y(final_data, 'mass', cat_names, None)

# %%
plt.show()