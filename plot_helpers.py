import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# base plotting function
def basic_plot(axs, data, xyz_names=[], split_name=None):

    # TODO: support z_name too?
    if len(xyz_names) == 2:
        # get x and y data
        x_data = data[xyz_names[0]]
        y_data = data[xyz_names[1]]
        # plot
        if split_name is not None:
            # split data by values in comp and plot
            splits = data[split_name].unique()
            n_splits = len(splits)
            s_nom = 100
            sizes = np.linspace(s_nom, 0.6*s_nom, n_splits)
            y_data_vars = []
            for value in splits:
                y_data_vars.append(np.var(y_data.loc[data[split_name]==value]))
            z_orders = np.argsort(y_data_vars)[::-1]
            sizes = [sizes[z] for z in z_orders]
            for idx, value in enumerate(splits):
                axs.scatter(x_data.loc[data[split_name]==value], y_data.loc[data[split_name]==value], s=sizes[idx], zorder = z_orders[idx], alpha=0.5, label=split_name+'='+str(value))
        else:
            # single scatter plot
            axs.scatter(x_data, y_data)
        # set axis labels
        axs.set_xlabel(xyz_names[0])
        axs.set_ylabel(xyz_names[1])
    elif len(xyz_names) == 3:
        # get x and y data
        x_data = data[xyz_names[0]]
        y_data = data[xyz_names[1]]
        z_data = data[xyz_names[2]]
        # plot
        if split_name is not None:
            # split data by values in comp and plot
            for value in data[split_name].unique():
                axs.scatter(x_data.loc[data[split_name]==value], y_data.loc[data[split_name]==value], z_data.loc[data[split_name]==value], label=split_name+'='+str(value))
        else:
            # single scatter plot
            axs.scatter(x_data, y_data, z_data)
        # set axis labels
        axs.set_xlabel(xyz_names[0])
        axs.set_ylabel(xyz_names[1])
        axs.set_zlabel(xyz_names[2])

# function for single plot (x vs y)
def plot_one_x_one_y(data, x_name, y_name, split_name=None):

    # single plot
    fig = plt.figure(figsize=(8, 8))
    axs = fig.add_subplot()

    # plotting function
    basic_plot(axs, data, [x_name, y_name], split_name)

    axs.set_title(y_name + ' vs ' + x_name)
    axs.legend(loc='upper right')

# function for 3d scatter plot (x,y vs z)
def plot_two_x_one_y(data, x_name, y_name, z_name, split_name=None):

    # single plot
    fig = plt.figure(figsize=(8, 8))
    axs = fig.add_subplot(projection='3d')

    # plotting function
    basic_plot(axs, data, [x_name, y_name, z_name], split_name)

    axs.set_title(x_name + ' vs ' + y_name + ' vs ' + z_name)
    axs.legend()

# function for generating plots of a single series of data vs. all simulation parameters
def plot_one_y_for_many_x(data, y_name, x_names, split_name=None):

    # based on length of param names, set subplots
    num_plots = len(x_names)
    r = num_plots//3
    if num_plots%3 != 0:
        r += 1
    c = min(num_plots, 3)
    if num_plots==4: c = 2
    figw = 6*c
    figh = 6*r

    # create figure
    fig, axs = plt.subplots(r,c, figsize=(figw, figh))
    fig.suptitle(y_name)

    for idx, ax in enumerate(axs.ravel()):
        if idx < num_plots:
            basic_plot(ax, data, [x_names[idx], y_name], split_name)
            ax.set_title(x_names[idx])
            if idx==0:
                ax.legend(loc='upper right')
        else:
            break

def plot_one_x_for_many_y(data, x_name, y_names, split_name=None):

    # based on length of cat names, set subplots
    num_plots = len(y_names)
    r = num_plots//3
    if num_plots%3 != 0:
        r += 1
    c = min(num_plots, 3)
    if num_plots==4: c = 2
    figw = 6*c
    figh = 6*r

    # create figure
    fig, axs = plt.subplots(r,c, figsize=(figw, figh))
    fig.suptitle(x_name)

    for idx, ax in enumerate(axs.ravel()):
        if idx < num_plots:
            basic_plot(ax, data, [x_name, y_names[idx]], split_name)
            ax.set_title(y_names[idx])
            if idx==0:
                ax.legend(loc='upper right')
        else:
            break

def plot_one_x_one_y_many_splits(data, x_name, y_name, split_names):

    # based on length of split names, set subplots
    num_plots = len(split_names)
    r = num_plots//3
    if num_plots%3 != 0:
        r += 1
    c = min(num_plots, 3)
    if num_plots==4: c = 2
    figw = 6*c
    figh = 6*r

    # create figure
    fig, axs = plt.subplots(r,c, figsize=(figw, figh))
    fig.suptitle(x_name + ' vs ' + y_name)

    for idx, ax in enumerate(axs.ravel()):
        if idx < num_plots:
            basic_plot(ax, data, [x_name, y_name], split_names[idx])
            ax.set_title(split_names[idx])
            ax.legend(loc='upper right')
        else:
            break