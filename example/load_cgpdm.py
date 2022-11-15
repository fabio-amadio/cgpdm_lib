#!/usr/bin/env python

# Copyright 2022 by Fabio Amadio.
# All rights reserved.
# This file is part of the cgpdm_lib,
# and is released under the "GNU General Public License".
# Please see the LICENSE file included in the package.

import torch
import numpy as np
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from sklearn.decomposition import PCA
from torch.distributions.normal import Normal
import pickle
import argparse
import time
from cgpdm_lib.models import CGPDM


# file parameters
p = argparse.ArgumentParser('load cgpdm')
p.add_argument('-model_name',
               type = str,
               default = 'cgpdm',
               help = 'name of the model backup files')

# load parameters
locals().update(vars(p.parse_known_args()[0]))

# load load_folder
load_folder = 'ROLLOUT/'

# load model config_dict and state_dict
config_dict = pickle.load(open(load_folder+model_name+'_config_dict.pt', 'rb'))
state_dict = torch.load(load_folder+model_name+'_state_dict.pt')

print('\nLOAD CGPDM:')
print(' - model name: '+model_name)
print(' - config_dict path: '+load_folder+model_name+'_config_dict.pt')
print(' - state_dict path: '+load_folder+model_name+'_state_dict.pt')

# define torch device and data type
dtype=torch.float64
device=torch.device('cpu')

# load config params
D = config_dict['D']
d = config_dict['d']
u_dim = config_dict['u_dim']
dyn_target = config_dict['dyn_target']
dyn_back_step = config_dict['dyn_back_step']
sigma_n_num_Y = config_dict['sigma_n_num_Y']
sigma_n_num_X = config_dict['sigma_n_num_X']

param_dict = {}
param_dict['D'] = D
param_dict['d'] = d
param_dict['u_dim'] = u_dim
param_dict['y_lambdas_init'] = np.ones(D)
param_dict['y_lengthscales_init'] = np.ones(d)
param_dict['y_sigma_n_init'] = np.ones(1)
param_dict['x_lambdas_init'] = np.ones(d)
param_dict['x_lengthscales_init'] = np.ones((d+u_dim)*dyn_back_step)
param_dict['x_sigma_n_init'] = np.ones(1)
param_dict['x_lin_coeff_init'] = np.ones((d+u_dim)*dyn_back_step+1)
param_dict['dyn_target'] = dyn_target
param_dict['dyn_back_step'] = dyn_back_step
param_dict['sigma_n_num_Y'] = sigma_n_num_Y
param_dict['sigma_n_num_X'] = sigma_n_num_X
param_dict['device'] = device

# model configuration and params loading
model = CGPDM(**param_dict)
model.load(config_dict, state_dict)


# get latent states
X_list = model.get_latent_sequences()

# latent trajectories
plt.figure()
plt.suptitle('Latent trajectories')
for j in range(d):
    plt.subplot(d,1,j+1)
    plt.ylabel(r'$x_{'+str(j+1)+'}$')
    for i in range(len(X_list)):
        plt.plot(X_list[i][:,j])
    plt.grid()
plt.show()


# Rollout generation on train data
for i in range(len(model.observations_list)):
    Y_true = model.observations_list[i]
    U = model.controls_list[i]
    X0 = X_list[0][0,:]

    U = torch.tensor(U, dtype=dtype, device=device)
    N = U.shape[0]

    Xhat, Yhat = model.rollout(num_steps=N, control_actions=U, X0=X0, flg_sample_X=False, flg_sample_Y=False, flg_noise=False)

    Yhat = Yhat.reshape(len(Yhat),64,3)
    Y_true = Y_true.reshape(len(Y_true),64,3)

    # 3D plot
    fig = plt.figure(i)
    ax = plt.axes(projection='3d')
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.5,1.5])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.5, 1.5])
    ax.set_zlabel('Z')
    ax.set_title('Train trajectory #'+str(i+1))
    scat_test = ax.plot(Y_true[0,:,0],Y_true[0,:,1],Y_true[0,:,2],'bo', ms = 2)[0]
    scat_hat = ax.plot(Yhat[0,:,0],Yhat[0,:,1],Yhat[0,:,2],'ro', ms = 2)[0]
    plt.legend([r'$\mathbf{y}$', r'$\hat{\mathbf{y}}$'])

    def plotter(k, scat_test, Y_true, scat_hat, Yhat):
        scat_test.set_data(np.array([Y_true[k,:,0],Y_true[k,:,1]]))
        scat_test.set_3d_properties(Y_true[k,:,2])
        scat_hat.set_data(np.array([Yhat[k,:,0],Yhat[k,:,1]]))
        scat_hat.set_3d_properties(Yhat[k,:,2])

    animator = ani.FuncAnimation(fig, plotter, len(Y_true), fargs=(scat_test, Y_true, scat_hat, Yhat), interval = 20, repeat=False)

    plt.show()
