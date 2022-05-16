# IMPORT PACKAGES
import numpy as np
import pandas as pd
import scipy
import scipy.io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pickle
from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import json

from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# FUNCTIONS DEFINITION
def simulate_vehicle_motion(X_true,
                            Y_true,
                            Y_pred,
                            means,
                            standards,
                            extra_variables,
                            extra_variables_names,
                            prediction_features,
                            config_parameters,
                            scene):
    present = X_true.shape[1] - 1
    X_true2 = X_true[scene, -1, :] * standards + means
    Y_true2 = Y_true[scene, :, :] * standards + means
    Y_pred2 = Y_pred[scene, :, :] * standards + means
    extra_var = extra_variables[scene, present, :]

    initial_time = 0
    
    relativeDistance = X_true2[prediction_features.index('relativeDistance')]
    relativeYaw = np.deg2rad(X_true2[prediction_features.index('relativeYaw')])
    x_centerline = extra_var[len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('centerlineX')]
    y_centerline = extra_var[len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('centerlineY')]
    x_normal = extra_var[len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('normalX')]
    y_normal = extra_var[len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('normalY')]
    vx = X_true2[prediction_features.index('chassis_velocities.longitudinal')]
    vy = -X_true2[prediction_features.index('chassis_velocities.lateral')]
    tangentX = extra_var[len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('tangentX')]
    tangentY = extra_var[len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('tangentY')]
    road_angle = np.arctan2(tangentY, tangentX)
    x_true = extra_var[extra_variables_names['vehicle'].index('chassis_displacements.longitudinal')]
    y_true = extra_var[extra_variables_names['vehicle'].index('chassis_displacements.lateral')]

    state = list()
    state.append(x_true)  # initial X position
    state.append(y_true)  # initial Y position
    state.append(vx)  # initial Vx
    state.append(vy)  # initial Vy
    state.append(relativeYaw + road_angle)  # initial global yaw
    state = np.array(state)

    controls = Y_pred2[:, [prediction_features.index('chassis_accelerations.longitudinal'),
                           prediction_features.index('chassis_accelerations.lateral'),
                           prediction_features.index('chassis_velocities.yaw')]]
    real_controls = Y_true2[:, [prediction_features.index('chassis_accelerations.longitudinal'),
                                prediction_features.index('chassis_accelerations.lateral'),
                                prediction_features.index('chassis_velocities.yaw')]]
    controls[:, 1] = -controls[:, 1]
    real_controls[:, 1] = -real_controls[:, 1]
    controls[:, 2] = np.deg2rad(controls[:, 2])
    real_controls[:, 2] = np.deg2rad(real_controls[:, 2])

    def f(y, t, u):
        times = np.arange(0, 0.1 * config_parameters['tF'], 0.1)
        # time_index = np.nonzero(times <= t)[0][-1]
        interpol = interp1d(times, u.T, kind='previous', fill_value=(u[0, :], u[-1, :]),
                                              bounds_error=False)
        u_new = interpol(t)
        out = np.zeros(y.shape)
        beta = np.arctan(y[3] / y[2])
        vel = np.sqrt((y[2] ** 2 + y[3] ** 2))
        out[0] = vel * np.cos(beta + y[4])
        out[1] = vel * np.sin(beta + y[4])
        out[2] = u_new[0] + u_new[2] * y[3]
        out[3] = u_new[1] - u_new[2] * y[2]
        out[4] = u_new[2]

        return out

    lower_sampling = 0.1
    t_eval = np.arange(0, 0.1 * config_parameters['tF'] + lower_sampling, lower_sampling)
    solution_pred = odeint(lambda y, t: f(y, t, controls), state,
                                           t=t_eval)  # , atol=1e-9, rtol=1e-9, hmax=0.001)
    solution_true = odeint(lambda y, t: f(y, t, real_controls), state,
                                           t=t_eval)  # , atol=1e-9, rtol=1e-9, hmax=0.001)

    solution_true = solution_true.T
    solution_pred = solution_pred.T

    return present, solution_true, solution_pred

def simulate_road_margins(X_true2,
                          extra_variables,
                          extra_variables_names,
                          present,
                          scene):

    leftMarginX = extra_variables[scene, present:,
                  len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('leftMarginX')]
    leftMarginY = extra_variables[scene, present:,
                  len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('leftMarginY')]
    rightMarginX = extra_variables[scene, present:,
                   len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('rightMarginX')]
    rightMarginY = extra_variables[scene, present:,
                   len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('rightMarginY')]
    # x_true = extra_variables[scene, present:, extra_variables_names['vehicle'].index('chassis_displacements.longitudinal')]
    # y_true = extra_variables[scene, present:, extra_variables_names['vehicle'].index('chassis_displacements.lateral')]
    # tangentX = extra_variables[scene, present:, len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('tangentX')]
    # tangentY = extra_variables[scene, present:, len(extra_variables_names['vehicle']) + extra_variables_names['road'].index('tangentY')]
    # road_angle = np.arctan2(tangentY, tangentX)
    # relativeYaw = np.hstack([np.deg2rad(X_true2[prediction_features.index('relativeYaw')]), np.deg2rad(Y_true2[:, prediction_features.index('relativeYaw')])])
    # global_orientation = road_angle + relativeYaw

    return leftMarginX, leftMarginY, rightMarginX, rightMarginY


def plot_driving_scenario(fig,
                          ax,
                          leftMarginX,
                          leftMarginY,
                          rightMarginX,
                          rightMarginY,
                          solution_true,
                          solution_pred,
                          cars_size=0.005,
                          label_size = 6,
                          tick_size=6,
                          legend_size=8):

    ax.plot(leftMarginX, leftMarginY, linewidth=2, color='k')
    ax.plot(rightMarginX, rightMarginY, linewidth=2, color='k')

    ax.plot(solution_true[0, :], solution_true[1, :], linewidth=1.4, color='green', alpha=0.4)
    ax.plot(solution_pred[0, :], solution_pred[1, :], linewidth=1.4, color='tab:blue')
    # ax.plot(x_true, y_true, '--', color='red', alpha=0.3)

    car_pred = plt.imread('icons/Car TOP_VIEW 375397.png')
    car_true = plt.imread('icons/Car TOP_VIEW ABCB51.png')
    # car_gt = plt.imread('icons/Car TOP_VIEW F05F78.png')
    for time_sample in range(0, solution_true.shape[1], 5):
        r_img = rotate(car_pred, np.rad2deg(solution_pred[4, time_sample]),
                       reshape=True)
        oi = OffsetImage(r_img, zoom=cars_size, zorder=700)
        veh_box = AnnotationBbox(oi, (solution_pred[0, time_sample], solution_pred[1, time_sample]), frameon=False)
        veh_box.zorder = 700
        ax.add_artist(veh_box)

        r_img = rotate(car_true, np.rad2deg(solution_true[4, time_sample]),
                       reshape=True)
        oi = OffsetImage(r_img, zoom=cars_size, zorder=700)
        veh_box = AnnotationBbox(oi, (solution_true[0, time_sample], solution_true[1, time_sample]), frameon=False)
        veh_box.zorder = 700
        ax.add_artist(veh_box)

        # r_img = rotate(car_gt, np.rad2deg(global_orientation[time_sample]),
        #                reshape=True)
        # oi = OffsetImage(r_img, zoom=0.005, zorder=700)
        # veh_box = AnnotationBbox(oi, (x_true[time_sample], y_true[time_sample]), frameon=False)
        # veh_box.zorder = 700
        # ax.add_artist(veh_box)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]', fontsize=label_size)
    ax.set_ylabel('Y [m]', fontsize=label_size)

    ax.plot([], [], c='green', linewidth=1.4, alpha=0.4, label='Truth')
    ax.plot([], [], c='blue', linewidth=1.4, label='Prediction')
    ax.plot([], [], c='k', linewidth=2, label='Road')
    ax.legend(loc='best', fontsize=legend_size)

    # current_ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)
    ax.tick_params(axis='both', which='both', labelsize=tick_size)

    return fig, ax

def plot_whole_set_prediction(fig,
                              ax,
                              q,
                              X_true,
                              Y_true,
                              Y_pred,
                              means,
                              standards,
                              sampling_time,
                              names):

    for j in range(q):
        # if j != 1:
        #     continue
        # else:
        #     current_ax = ax

        current_ax = ax[j]

        X_true2 = X_true[:, -1, j] * standards[j] + means[j]
        Y_true2 = Y_true[:, -1, j] * standards[j] + means[j]
        Y_pred2 = Y_pred[:, -1, j] * standards[j] + means[j]

        current_ax.plot(sampling_time * (X_true.shape[1] + np.arange(Y_true2.shape[0])), Y_true2, linewidth=0.8,
                        color='green', alpha=0.4)
        current_ax.plot(sampling_time * (X_true.shape[1] + np.arange(Y_pred2.shape[0])), Y_pred2, linewidth=0.8,
                        color='tab:blue', alpha=0.9)

        # current_ax.legend(loc='best', labels=["Truth", "Prediction"], fontsize=6)
        # current_ax.set_xlabel("Time steps", fontsize=6)
        # current_ax.set_xlabel("Time [s]", fontsize=6)
        if j == q - 1:
            current_ax.legend(labels=["Truth", "Prediction"], fontsize=6)
            current_ax.set_xlabel("Time steps", fontsize=6)
            current_ax.set_xlabel("Time [s]", fontsize=6)

        current_ax.set_ylabel(names[j], fontsize=6)
        current_ax.tick_params(axis='both', which='both', labelsize=6)

        current_ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)

    return fig, ax

def plot_horizons(fig,
                  ax,
                  q,
                  X_true,
                  Y_true,
                  Y_pred,
                  means,
                  standards,
                  sampling_time,
                  scene,
                  names,
                  label_size=6,
                  tick_size=6,
                  legend_size=8):


    for j in range(q):
        current_ax = ax[j]

        X_true2 = X_true[scene, :, j] * standards[j] + means[j]
        Y_true2 = Y_true[scene, :, j] * standards[j] + means[j]
        Y_pred2 = Y_pred[scene, :, j] * standards[j] + means[j]
        current_ax.plot(sampling_time * np.arange(X_true.shape[1]), X_true2, linewidth=1.4, linestyle='--',
                        color='tab:grey')
        current_ax.plot(sampling_time * (X_true.shape[1] + np.arange(Y_true.shape[1])), Y_true2, linewidth=1.4,
                        color='green', alpha=0.4)
        current_ax.plot(sampling_time * (X_true.shape[1] + np.arange(Y_pred.shape[1])), Y_pred2, linewidth=1.4,
                        color='tab:blue')

        # if (index==0 and j==q-1):
        if j == 0:
            current_ax.legend(labels=["Past", "Truth", "Prediction"], fontsize=legend_size)
        if j == q - 1:
            current_ax.set_xlabel("Time [s]", fontsize=label_size)

        current_ax.set_ylabel(names[j], fontsize=label_size)
        current_ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)
        current_ax.tick_params(axis='both', which='both', labelsize=tick_size)

    return fig, ax
