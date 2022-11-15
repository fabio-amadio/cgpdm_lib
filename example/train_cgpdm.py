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
p = argparse.ArgumentParser('train cgpdm')
p.add_argument('-seed',
               type = int,
               default = 1,
               help = 'seed')
p.add_argument('-num_data',
               type = int,
               default = 10,
               help = 'num_data')
p.add_argument('-deg',
               type = int,
               default = 30,
               help = 'oscillation angle')
p.add_argument('-d',
               type = int,
               default = 3,
               help = 'latent dim.')
p.add_argument('-num_opt_steps',
               type = int,
               default = 26,
               help = 'optimization steps')
p.add_argument('-lr',
               type = float,
               default = 0.01,
               help = 'learning rate')
p.add_argument('-flg_show',
               type = bool,
               default = False,
               help = 'show plots and video')

# load parameters
locals().update(vars(p.parse_known_args()[0]))

print('\nTRAIN CGPDM:')
print(' - random seed: '+str(seed))
print(' - training trajectories: '+str(num_data))
print(' - oscillation angle: '+str(deg))
print(' - latent dimension: '+str(d))
print(' - num. opt. steps: '+str(num_opt_steps))
print(' - learning rate: '+str(lr))
print(' - show results: '+str(flg_show))

# define torch device and data type
dtype=torch.float64
device=torch.device('cpu')

# load observation data
folder = 'DATA/8x8_rng_swing_'+str(deg)+'_deg/'
print('\nDATA FOLDER: '+folder)

Y_data = []
U_data = []
for i in range(41):
    Y_data.append(np.loadtxt(folder+'state_samples_rng_swing_'+str(i)+'.csv', delimiter=','))
    U_data.append(np.loadtxt(folder+'input_samples_rng_swing_'+str(i)+'.csv', delimiter=','))

# load test data
Y_test = []
U_test = []
for i in range(40,50):
    Y_test.append(np.loadtxt(folder+'state_samples_rng_swing_'+str(i)+'.csv', delimiter=','))
    U_test.append(np.loadtxt(folder+'input_samples_rng_swing_'+str(i)+'.csv', delimiter=','))


# init GPDM object
D = Y_data[0].shape[1]
u_dim = U_data[0].shape[1]
print('D', D)
print('u_dim', u_dim)
print('d', d)
dyn_target = 'delta' # 'full' or delta' 
dyn_back_step = 1 # 1 or 2

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
param_dict['sigma_n_num_Y'] = 10**-3
param_dict['sigma_n_num_X'] = 10**-3
param_dict['device'] = device
model = CGPDM(**param_dict)

# add training data
np.random.seed(seed)
training_set = np.random.randint(low=0, high=len(Y_data), size=num_data)
print('training set: ', training_set)
for i in training_set:
    model.add_data(Y_data[i],U_data[i])

# init model
model.init_X()
X_list_pca = model.get_latent_sequences()

# train model
start_time = time.time()
loss = model.train_lbfgs(num_opt_steps=num_opt_steps, num_print_steps=1, lr=lr)
end_time = time.time()
train_time = end_time - start_time
print("\nTotal Training Time: "+str(train_time)+" [s]")


# save model
save_folder = 'ROLLOUT/'
config_dict_path = save_folder+'cgpdm_config_dict.pt'
state_dict_path = save_folder+'cgpdm_state_dict.pt'
model.save(config_dict_path, state_dict_path)
np.savetxt(save_folder+'cgpdm_training_set.csv', np.array(training_set), delimiter=',')


# get latent states
X_list = model.get_latent_sequences()


if flg_show:
    # plot loss
    plt.figure()
    plt.plot(loss)
    plt.grid()
    plt.title('Loss')
    plt.xlabel('Optimization steps')
    plt.show()

    # latent trajectories
    plt.figure()
    plt.suptitle('Latent trajectories')
    for j in range(d):
        plt.subplot(d,1,j+1)
        plt.xlabel('Time [s]')
        plt.ylabel(r'$x_{'+str(j+1)+'}$')
        for i in range(len(X_list)):
            plt.plot(X_list[i][:,j])
        plt.grid()
    plt.show()


# Rollout generation on test data

# get initial latent state
X0_sum = X_list[0][0,:]
for n in range(1,len(X_list)):
    X0_sum = X0_sum + X_list[n][0,:]
X0 = X0_sum/len(X_list)

for i in range(len(Y_test)):
    Ytest = Y_test[i]
    Utest = U_test[i]

    Utest = torch.tensor(Utest, dtype=dtype, device=device)
    N = Utest.shape[0]

    Xhat, Yhat = model.rollout(num_steps=N, control_actions=Utest, X0=X0, flg_sample_X=False, flg_sample_Y=False, flg_noise=False)

    np.savetxt(save_folder+'test_samples_hat_'+str(i)+'.csv', Yhat, delimiter=',')
    np.savetxt(save_folder+'test_samples_'+str(i)+'.csv', Ytest, delimiter=',')


if flg_show:
    for i in range(len(Y_test)):
        Yhat = Yhat.reshape(len(Yhat),64,3)
        Ytest = Ytest.reshape(len(Ytest),64,3)

        # 3D plot
        fig = plt.figure(i)
        ax = plt.axes(projection='3d')
        ax.set_xlim3d([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylim3d([-1.5,1.5])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-1.5, 1.5])
        ax.set_zlabel('Z')
        ax.set_title('Test trajectory #'+str(i+1))
        scat_test = ax.plot(Ytest[0,:,0],Ytest[0,:,1],Ytest[0,:,2],'bo', ms = 2)[0]
        scat_hat = ax.plot(Yhat[0,:,0],Yhat[0,:,1],Yhat[0,:,2],'ro', ms = 2)[0]
        plt.legend([r'$\mathbf{y}$', r'$\hat{\mathbf{y}}$'])

        def plotter(k, scat_test, Ytest, scat_hat, Yhat):
            scat_test.set_data(np.array([Ytest[k,:,0],Ytest[k,:,1]]))
            scat_test.set_3d_properties(Ytest[k,:,2])
            scat_hat.set_data(np.array([Yhat[k,:,0],Yhat[k,:,1]]))
            scat_hat.set_3d_properties(Yhat[k,:,2])

        animator = ani.FuncAnimation(fig, plotter, len(Ytest), fargs=(scat_test, Ytest, scat_hat, Yhat), interval = 20, repeat=False)

        plt.show()
