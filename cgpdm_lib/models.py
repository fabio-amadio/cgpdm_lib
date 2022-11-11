""" 
Author: Fabio Amadio (fabioamadio93@gmail.com)
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.distributions.normal import Normal
import pickle
import sys
import pdb


class GPDM(torch.nn.Module):
    """
    Gaussian Process Dynamical Model

    Attributes
    ----------
    dtype : torch.dtype
        data type of torch tensors

    device : torch.device
        device on which a tensors will be allocated

    D : int
        observation space dimension

    d : int
        desired latent space dimension

    dyn_target : string
        dynamic map function target ('full' or 'delta')

    dyn_step_back : int
        memory of dynamic map function (1 or 2)

    y_log_lengthscales : torch.nn.Parameter
        log(lengthscales) of Y GP kernel

    y_log_lambdas : torch.nn.Parameter
        log(signal inverse std) of Y GP kernel

    y_log_sigma_n : torch.nn.Parameter
        log(noise std) of Y GP kernel

    x_log_lengthscales : torch.nn.Parameter
        log(lengthscales) of X GP kernel

    x_log_lambdas : torch.nn.Parameter
        log(signal inverse std) of X GP kernel

    x_log_sigma_n : torch.nn.Parameter
        log(noise std) of X GP kernel

    x_log_lin_coeff : torch.nn.Parameter
        log(linear coefficients) of X GP kernel

    X : torch.nn.Parameter
        latent states

    sigma_n_num_X : double 
        additional noise std for numerical issues in X GP

    sigma_n_num_Y : double
        additional noise std for numerical issues in Y GP

    observations_list : list(double)
        list of observation sequences   

    """
    def __init__(self, D, d, dyn_target, dyn_back_step,
                 y_lambdas_init, y_lengthscales_init, y_sigma_n_init,
                 x_lambdas_init, x_lengthscales_init, x_sigma_n_init, x_lin_coeff_init, 
                 flg_train_y_lambdas=True, flg_train_y_lengthscales=True, flg_train_y_sigma_n=True,
                 flg_train_x_lambdas=True, flg_train_x_lengthscales=True, flg_train_x_sigma_n=True, flg_train_x_lin_coeff=True,
                 sigma_n_num_Y=0., sigma_n_num_X=0., dtype=torch.float64, device=torch.device('cpu')):
        """
        Parameters
        ----------
        D : int
            observation space dimension

        d : int
            latent space dimension

        dyn_target : string
            dynamic map function target ('full' or 'delta')
        
        dyn_back_step : int
            memory of dynamic map function (1 or 2)

        y_lambdas_init : double
            initial signal std parameter for GP Y (dimension: D)

        y_lengthscales_init : double
            initial lengthscales parameter for GP Y (dimension: d)

        y_sigma_n_init : double
            initial noise std parameter for GP Y (dimension: 1)
        
        x_lambdas_init : double
            initial signal std parameter for GP X (dimension: d)

        x_lengthscales_init : double
            initial lengthscales parameter for GP X (dimension: d*dyn_back_step)

        x_sigma_n_init : double
            initial noise std parameter for GP X (dimension: 1)

        x_lin_coeff_init : double
            initial linear coefficients for GP X (dimension: d*dyn_back_step+1)

        flg_train_y_lambdas : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_y_lengthscales : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_y_sigma_n : boolean (optional)
            requires_grad flag for associated parameter
                 
        flg_train_x_lambdas : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_x_lengthscales : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_x_sigma_n : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_x_lin_coeff : boolean (optional)
            requires_grad flag for associated parameter

        sigma_n_num_Y : double (optional)
            additional noise std for numerical issues in X GP

        sigma_n_num_X : double (optional)
            additional noise std for numerical issues in X GP

        dtype: torch.dtype (optional)
            data type of torch tensors

        device: torch.device (optional)
            device on which a tensors will be allocated

        """
        super(GPDM,self).__init__()

        # torch parameters
        self.dtype = dtype
        self.device = device

        # observation dimension
        self.D = D
        # desired latent dimension
        self.d = d
        # dynamic model target
        self.dyn_target = dyn_target
        # dynamic model input
        self.dyn_back_step = dyn_back_step

        # Set Y-kernel parameters
        self.y_log_lengthscales = torch.nn.Parameter(torch.tensor(np.log(np.abs(y_lengthscales_init)),
                                                                  dtype=self.dtype,
                                                                  device=self.device),
                                                                  requires_grad=flg_train_y_lengthscales)
        self.y_log_lambdas = torch.nn.Parameter(torch.tensor(np.log(np.abs(y_lambdas_init)),
                                                             dtype=self.dtype,
                                                             device=self.device),
                                                             requires_grad=flg_train_y_lambdas)
        self.y_log_sigma_n = torch.nn.Parameter(torch.tensor(np.log(np.abs(y_sigma_n_init)),
                                                             dtype=self.dtype,
                                                             device=self.device),
                                                             requires_grad=flg_train_y_sigma_n)
        # Set X-kernel parameters
        self.x_log_lengthscales = torch.nn.Parameter(torch.tensor(np.log(np.abs(x_lengthscales_init)),
                                                                  dtype=self.dtype,
                                                                  device=self.device),
                                                                  requires_grad=flg_train_x_lengthscales)
        self.x_log_lambdas = torch.nn.Parameter(torch.tensor(np.log(np.abs(x_lambdas_init)),
                                                             dtype=self.dtype,
                                                             device=self.device),
                                                             requires_grad=flg_train_x_lambdas)
        self.x_log_sigma_n = torch.nn.Parameter(torch.tensor(np.log(np.abs(x_sigma_n_init)),
                                                             dtype=self.dtype,
                                                             device=self.device),
                                                             requires_grad=flg_train_x_sigma_n)
        self.x_log_lin_coeff = torch.nn.Parameter(torch.tensor(np.log(np.abs(x_lin_coeff_init)),
                                                               dtype=self.dtype,
                                                               device=self.device),
                                                               requires_grad=flg_train_x_lin_coeff)
        # additional noise variance for numerical issues
        self.sigma_n_num_Y = sigma_n_num_Y
        self.sigma_n_num_X = sigma_n_num_X

        # Initialize observations
        self.observations_list = []
        self.num_sequences = 0

    def set_evaluation_mode(self):
        """
        Set the model in evaluation mode
        """
        self.flg_trainable_list = []
        for p in self.parameters():
            p.requires_grad = False

    def set_training_mode(self, model='all'):
        """
        Set the model in training mode
        
        Parameters
        ----------

        model : string ['all', 'latent' or 'dynamics'] (optional)
            'all' set all requires_grad to True
            'latent' set only GP Y parameters to True
            'dynamics' set only GP X parameters to True
        """
        if model == 'all':
            for i, p in enumerate(self.parameters()):
                p.requires_grad = True
        elif model == 'latent':
            self.y_log_lengthscales.requires_grad = True
            self.y_log_lambdas.requires_grad = True
            self.y_log_sigma_n.requires_grad = True
            self.x_log_lengthscales.requires_grad = False
            self.x_log_lambdas.requires_grad = False
            self.x_log_sigma_n.requires_grad = False
            self.x_log_lin_coeff.requires_grad = False
        elif model == 'dynamics':
            self.y_log_lengthscales.requires_grad = False
            self.y_log_lambdas.requires_grad = False
            self.y_log_sigma_n.requires_grad = False
            self.x_log_lengthscales.requires_grad = True
            self.x_log_lambdas.requires_grad = True
            self.x_log_sigma_n.requires_grad = True
            self.x_log_lin_coeff.requires_grad = True
        else:
            raise ValueError('model must be \'all\', \'latent\' or \'dynamics\'!')      

    def add_data(self, Y):
        """
        Add observation data to self.observations_list

        Parameters
        ----------

        Y : double
            observation data (dimension: N x D)

        """
        if Y.shape[1]!=self.D:
            raise ValueError('Y must be a N x D matrix collecting observation data!')

        self.observations_list.append(Y)
        self.num_sequences = self.num_sequences+1

        print('Num. of sequences = '+str(self.num_sequences)+' [Data points = '+str(np.concatenate(self.observations_list, 0).shape[0])+']')

    def get_y_kernel(self, X1, X2, flg_noise=True):
        """
        Compute the latent mapping kernel (GP Y)
        
        Parameters
        ----------

        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        flg_noise : boolean (optional)
            add noise to kernel matrix

        Return
        ------
        K_y(X1,X2)

        """
        return self.get_rbf_kernel(X1, X2, self.y_log_lengthscales, self.y_log_sigma_n, self.sigma_n_num_Y, flg_noise)

    def get_x_kernel(self, X1, X2, flg_noise=True):
        """
        Compute the latent dynamic kernel (GP X)

        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        flg_noise : boolean (optional)
            add noise to kernel matrix

        Return
        ------
        K_x(X1,X2) 

        """
        return self.get_rbf_kernel(X1, X2, self.x_log_lengthscales, self.x_log_sigma_n, self.sigma_n_num_X, flg_noise) + \
               self.get_lin_kernel(X1, X2, self.x_log_lin_coeff)   

    def get_rbf_kernel(self, X1, X2, log_lengthscales_par, log_sigma_n_par, sigma_n_num=0, flg_noise=True):
        """
        Compute RBF kernel on X1, X2 points (without considering signal variance)

        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        log_lengthscales_par : tensor(dtype)
            log(lengthscales) RBF kernel

        log_sigma_n_par : tensor(dtype)
            log(noise std)  RBF kernel

        sigma_n_num : double
            additional noise std for numerical issues

        flg_noise : boolean (optional)
            add noise to kernel matrix

        Return
        ------
        K_rbf(X1,X2)

        """

        if flg_noise:
            N = X1.shape[0]
            return torch.exp(-self.get_weighted_distances(X1, X2, log_lengthscales_par)) + \
                torch.exp(log_sigma_n_par)**2*torch.eye(N, dtype=self.dtype, device=self.device) + sigma_n_num**2*torch.eye(N, dtype=self.dtype, device=self.device)

        else:
            return torch.exp(-self.get_weighted_distances(X1, X2, log_lengthscales_par))

    def get_weighted_distances(self, X1, X2, log_lengthscales_par):
        """
        Computes (X1-X2)^T*Sigma^-2*(X1-X2) where Sigma = diag(exp(log_lengthscales_par))

        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        log_lengthscales_par : tensor(dtype)
            log(lengthscales)
        
        Return
        ------
        dist = (X1-X2)^T*Sigma^-2*(X1-X2)

        """
        lengthscales = torch.exp(log_lengthscales_par)

        X1_sliced = X1/lengthscales
        X1_squared = torch.sum(X1_sliced.mul(X1_sliced), dim=1, keepdim=True)
        X2_sliced = X2/lengthscales
        X2_squared = torch.sum(X2_sliced.mul(X2_sliced), dim=1, keepdim=True)
        dist = X1_squared + X2_squared.transpose(dim0=0, dim1=1) -2*torch.matmul(X1_sliced,X2_sliced.transpose(dim0=0, dim1=1))

        return dist


    def get_lin_kernel(self, X1, X2, log_lin_coeff_par):
        """
        Computes linear kernel on X1, X2 points: [X1,1]^T*Sigma*[X2,1] where Sigma=diag(exp(log_lin_coeff_par)) 
        
        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        log_lin_coeff_par : tensor(dtype)
            log(linear coefficients)

        Return
        ------
        K_lin(X1,X2)

        """
        Sigma = torch.diag(torch.exp(log_lin_coeff_par)**2)
        X1 = torch.cat([X1,torch.ones(X1.shape[0],1, dtype=self.dtype, device=self.device)],1)
        X2 = torch.cat([X2,torch.ones(X2.shape[0],1, dtype=self.dtype, device=self.device)],1)
        return torch.matmul(X1, torch.matmul(Sigma, X2.transpose(0,1)))


    def get_y_neg_log_likelihood(self, Y, X, N):
        """
        Compute latent negative log-likelihood Ly
        
        Parameters
        ----------

        Y : tensor(dtype)
            observation matrix

        X : tensor(dtype)
            latent state matrix

        N : int
            number of data points
        
        Return
        ------
        L_y = D/2*log(|K_y(X,X)|) + 1/2*trace(K_y^-1*Y*W_y^2*Y) - N*log(|W_y|)

        """
        K_y = self.get_y_kernel(X,X)
        U, info = torch.linalg.cholesky_ex(K_y, upper=True)
        U_inv = torch.inverse(U)
        Ky_inv = torch.matmul(U_inv,U_inv.t())
        log_det_K_y = 2*torch.sum(torch.log(torch.diag(U)))

        W = torch.diag(torch.exp(self.y_log_lambdas))
        W2 = torch.diag(torch.exp(self.y_log_lambdas)**2)
        log_det_W = 2*torch.sum(self.y_log_lambdas)

        # return self.D/2*log_det_K_y+1/2*torch.trace(torch.chain_matmul(Ky_inv,Y,W2,Y.transpose(0,1)))-N*log_det_W
        return self.D/2*log_det_K_y+1/2*torch.trace(torch.linalg.multi_dot([Ky_inv,Y,W2,Y.transpose(0,1)]))-N*log_det_W

    def get_x_neg_log_likelihood(self, Xout, Xin, N):
        """
        Compute dynamics map negative log-likelihood Lx
        
        Parameters
        ----------

        Xout : tensor(dtype)
            dynamics map output matrix

        Xin : tensor(dtype)
            dynamics map input matrix

        N : int
            number of data points
        
        Return
        ------
        L_x = d/2*log(|K_x(Xin,Xin)|) + 1/2*trace(K_x^-1*Xout*W_x^2*Xout) - (N-dyn_back_step)*log(|W_x|)

        """
        
        K_x = self.get_x_kernel(Xin,Xin)
        U, info = torch.linalg.cholesky_ex(K_x, upper=True)
        U_inv = torch.inverse(U)
        Kx_inv = torch.matmul(U_inv,U_inv.t())
        log_det_K_x = 2*torch.sum(torch.log(torch.diag(U)))

        W = torch.diag(torch.exp(self.x_log_lambdas))
        W2 = torch.diag(torch.exp(self.x_log_lambdas)**2)
        log_det_W = 2*torch.sum(self.x_log_lambdas)

        # return self.d/2*log_det_K_x+1/2*torch.trace(torch.chain_matmul(Kx_inv,Xout,W2,Xout.transpose(0,1)))-Xin.shape[0]*log_det_W
        return self.d/2*log_det_K_x+1/2*torch.trace(torch.linalg.multi_dot([Kx_inv,Xout,W2,Xout.transpose(0,1)]))-Xin.shape[0]*log_det_W


    def get_Xin_Xout_matrices(self, X=None, target=None, back_step=None):
        """
        Compute GP input and output matrices (Xin, Xout) for GP X

        Parameters
        ----------

        X : tensor(dtype) (optional)
            latent state matrix

        target : string (optional)
            dynamic map function target ('full' or 'delta')
        
        back_step : int (optional)
            memory of dynamic map function (1 or 2)
        
        Return
        ------
        Xin : GP X input matrix

        Xout : GP X output matrix

        start_indeces : list of sequences' start indeces

        """
        if X==None:
            X = self.X
        if target==None:
            target = self.dyn_target
        if back_step==None:
            back_step = self.dyn_back_step

        X_list = []
        x_start_index = 0
        start_indeces = []
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(X[x_start_index:x_start_index+sequence_length,:])
            start_indeces.append(x_start_index)
            x_start_index = x_start_index+sequence_length           

        if target == 'full' and back_step == 1:
            # in: x(t)
            Xin = X_list[0][0:-1,:]
            # out: x(t+1)
            Xout = X_list[0][1:,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, X_list[j][0:-1,:]),0)
                Xout = torch.cat((Xout, X_list[j][1:,:]),0)

        elif target == 'full' and back_step == 2:
            # in: [x(t), x(t-1)]
            Xin = torch.cat((X_list[0][1:-1,:],X_list[0][0:-2,:]), 1)
            # out: x(t+1)
            Xout = X_list[0][2:,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][1:-1,:],X_list[j][0:-2,:]), 1)),0)
                Xout = torch.cat((Xout, X_list[j][2:,:]),0)

        elif target == 'delta' and back_step == 1:
            # in: x(t)
            Xin = X_list[0][0:-1,:]
            # out: x(t+1)-x(t)
            Xout = X_list[0][1:,:] - X_list[0][0:-1,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, X_list[j][0:-1,:]),0)
                Xout = torch.cat((Xout, X_list[j][1:,:] - X_list[j][0:-1,:]),0)

        elif target == 'delta' and back_step == 2:
            # in: [x(t), x(t-1)]
            Xin = torch.cat((X_list[0][1:-1,:],X_list[0][0:-2,:]), 1)
            # out: x(t+1)-x(t)
            Xout = X_list[0][2:,:] - X_list[0][1:-1,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][1:-1,:],X_list[j][0:-2,:]), 1)),0)
                Xout = torch.cat((Xout, X_list[j][2:,:] - X_list[j][1:-1,:]),0)
        
        else:
            raise ValueError('target must be either \'full\' or \'delta\' \n back_step must be either 1 or 2')

        return Xin, Xout, start_indeces


    def gpdm_loss(self, Y, N, balance = 1):
        """
        Calculate GPDM loss function L = Lx + B*Ly
        
        Parameters
        ----------

        Y : tensor(dtype)
            observation matrix

        X : tensor(dtype)
            latent state matrix

        N : int
            number of data points

        balance : double (optional)
            balance factor B
        
        Return
        ------
        GPDM loss = Ly + B*Lx

        """
        Xin, Xout, start_indeces = self.get_Xin_Xout_matrices()

        lossY = self.get_y_neg_log_likelihood(Y, self.X, N)
        lossX = self.get_x_neg_log_likelihood(Xout, Xin, N)

        loss = lossY + balance*lossX


        return loss


    def init_X(self):
        """
        Initalize latent variables matrix with PCA
        """
        Y = self.get_Y()
        pca = PCA(n_components=self.d)
        X0 = pca.fit_transform(Y)

        # set latent variables as parameters
        self.X = torch.nn.Parameter(torch.tensor(X0, dtype=self.dtype, device=self.device), requires_grad=True)

        # init inverse kernel matrices
        Ky = self.get_y_kernel(self.X,self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv,U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin,Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv,U_inv.t())

    def get_Y(self):
        """
        Create observation matrix Y from observations_list

        Return
        ------

        Y : observation matrix
        """

        observation = np.concatenate(self.observations_list, 0)
        # self.meanY = np.mean(observation,0)
        self.meanY = 0
        Y = observation-self.meanY
        return Y

    def train_adam(self, num_opt_steps, num_print_steps, lr=0.01, balance=1):
        """
        Optimize model with Adam

        Parameters
        ----------

        num_opt_steps : int 
            number of optimization steps

        num_print_steps : int
            number of steps between printing info

        lr : double
            learning rate

        balance : double
            balance factor for gpdm_loss

        Return
        ------

        losses : list of loss evaluated

        """

        print('\n### Model Training (Adam) ###')
        # create observation matrix
        Y = self.get_Y()
        N = Y.shape[0]
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device)

        self.set_training_mode('all')

        # define optimizer
        f_optim = lambda p : torch.optim.Adam(p, lr=lr)
        optimizer = f_optim(self.parameters())

        t_start = time.time()
        losses = []
        for epoch in range(num_opt_steps):
            optimizer.zero_grad()
            loss = self.gpdm_loss(Y, N, balance)
            loss.backward()
            if torch.isnan(loss):
                print('Loss is nan')
                break

            optimizer.step()

            losses.append(loss.item())

            if epoch % num_print_steps == 0:
                print('\nGPDM Opt. EPOCH:', epoch)
                print('Running loss:', "{:.4e}".format(loss.item()))
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                t_start = t_stop

        # save inverse kernel matrices after training
        Ky = self.get_y_kernel(self.X,self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv,U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin,Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv,U_inv.t())

        return losses

    def train_lbfgs(self, num_opt_steps, num_print_steps, lr=0.01, balance=1):
        """
        Optimize model with L-BFGS

        Parameters
        ----------

        num_opt_steps : int 
            number of optimization steps

        num_print_steps : int
            number of steps between printing info

        lr : double
            learning rate

        balance : double
            balance factor for gpdm_loss

        Return
        ------

        losses: list of loss evaluated

        """

        print('\n### Model Training (L-BFGS) ###')
        # create observation matrix
        Y = self.get_Y()
        N = Y.shape[0]
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        
        # define optimizer
        f_optim = lambda p : torch.optim.LBFGS(p, lr=lr, max_iter=20, history_size=7, line_search_fn='strong_wolfe')
        optimizer = f_optim(self.parameters())
        losses = []
        t_start = time.time()
        for epoch in range(num_opt_steps):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                loss = self.gpdm_loss(Y, N, balance)

                if loss.requires_grad:
                    loss.backward()
                return loss

            losses.append(closure().item())
            optimizer.step(closure)

            if epoch % num_print_steps == 0:
                print('\nGPDM Opt. EPOCH:', epoch)
                print('Running loss:', "{:.4e}".format(losses[-1]))
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                t_start = t_stop

        # save inverse kernel matrices after training
        Ky = self.get_y_kernel(self.X,self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv,U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin,Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv,U_inv.t())

        return losses

    def get_latent_sequences(self):
        """
        Return the latent trajectories associated to each observation sequence recorded

        Return
        ------

        X_list : list of latent states associated to each observation sequence
        """
        X_np = self.X.clone().detach().cpu().numpy()
        X_list = []
        x_start_index = 0
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(X_np[x_start_index:x_start_index+sequence_length,:])
            x_start_index = x_start_index+sequence_length

        return X_list

    def map_x_to_y(self, Xstar, flg_noise=False):
        """
        Map Xstar to observation space: return mean and variance
        
        Parameters
        ----------

        Xstar : tensor(dtype)
            input latent state matrix 

        flg_noise : boolean
            add noise to prediction variance

        Return
        ------

        mean_Y_pred : mean of Y prediction

        diag_var_Y_pred : variance of Y prediction


        """

        Y_obs = self.get_Y()
        Y_obs = torch.tensor(Y_obs, dtype=self.dtype, device=self.device)

        Ky_star = self.get_y_kernel(self.X,Xstar,False)

        # mean_Y_pred = torch.chain_matmul(Y_obs.t(),self.Ky_inv,Ky_star).t()
        mean_Y_pred = torch.linalg.multi_dot([Y_obs.t(),self.Ky_inv,Ky_star]).t()
        diag_var_Y_pred_common = self.get_y_diag_kernel(Xstar, flg_noise) - torch.sum(torch.matmul(Ky_star.t(),self.Ky_inv)*Ky_star.t(),dim=1)
        y_log_lambdas = torch.exp(self.y_log_lambdas)**-2
        diag_var_Y_pred = diag_var_Y_pred_common.unsqueeze(1)*y_log_lambdas.unsqueeze(0)

        return mean_Y_pred + torch.tensor(self.meanY, dtype=self.dtype, device=self.device), diag_var_Y_pred

    def get_y_diag_kernel(self, X, flg_noise=False):
        """
        Compute only the diagonal of the latent mapping kernel GP Y

        Parameters
        ----------

        X : tensor(dtype)
            latent state matrix

        flg_noise : boolean
            add noise to prediction variance

        Return
        ------
        GP Y diag covariance matrix

        """

        n = X.shape[0]
        if flg_noise:
            return torch.ones(n, dtype=self.dtype, device=self.device) + torch.exp(self.y_log_sigma_n)**2 + self.sigma_n_num_Y**2
        else:
            return torch.ones(n, dtype=self.dtype, device=self.device)

    def map_x_dynamics(self, Xstar, flg_noise=False):
        """
        Map Xstar to GP dynamics output

        Parameters
        ----------

        Xstar : tensor(dtype)
            input latent state matrix 

        flg_noise : boolean
            add noise to kernel matrix

        Return
        ------

        mean_Xout_pred : mean of Xout prediction

        diag_var_Xout_pred : variance of Xout prediction

        """
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        
        Kx_star = self.get_x_kernel(Xin,Xstar,False)
   
        # mean_Xout_pred = torch.chain_matmul(Xout.t(),self.Kx_inv,Kx_star).t()
        mean_Xout_pred = torch.linalg.multi_dot([Xout.t(),self.Kx_inv,Kx_star]).t()
        diag_var_Xout_pred_common = self.get_x_diag_kernel(Xstar, flg_noise) - torch.sum(torch.matmul(Kx_star.t(),self.Kx_inv)*Kx_star.t(),dim=1)
        x_log_lambdas = torch.exp(self.x_log_lambdas)**-2
        diag_var_Xout_pred = diag_var_Xout_pred_common.unsqueeze(1)*x_log_lambdas.unsqueeze(0)

        return mean_Xout_pred, diag_var_Xout_pred

    def get_x_diag_kernel(self, X, flg_noise=False):
        """
        Compute only the diagonal of the dynamics mapping kernel GP Y

        Parameters
        ----------

        X : tensor(dtype)
            latent state matrix

        flg_noise : boolean
            add noise to prediction variance
        
        Return
        ------
        GP X diag covariance matrix

        """

        n = X.shape[0]
        Sigma = torch.diag(torch.exp(self.x_log_lin_coeff)**2)
        X = torch.cat([X,torch.ones(X.shape[0],1, dtype=self.dtype, device=self.device)],1)
        if flg_noise:
            return torch.ones(n, dtype=self.dtype, device=self.device) + torch.exp(self.x_log_sigma_n)**2 + self.sigma_n_num_X**2 +\
                   torch.sum(torch.matmul(X, Sigma)*(X), dim=1)
        else:
            return torch.ones(n, dtype=self.dtype, device=self.device) + \
                   torch.sum(torch.matmul(X, Sigma)*(X), dim=1)

    def get_next_x(self, gp_mean_out, gp_out_var, Xold, flg_sample=False):
        """
        Predict GP X dynamics output to next latent state

        Parameters
        ----------

        gp_mean_out : tensor(dtype)
            mean of the GP X dynamics output

        gp_out_var : tensor(dtype)
            variance of the GP X dynamics output

        Xold : tensor(dtype)
            present latent state

        flg_noise : boolean
            add noise to prediction variance

        Return
        ------

        Predicted new latent state

        """

        distribution = Normal(gp_mean_out, torch.sqrt(gp_out_var))  
        if self.dyn_target == 'full':
            if flg_sample:
                return distribution.rsample()
            else:
                return gp_mean_out

        if self.dyn_target == 'delta':
            if flg_sample:
                return Xold + distribution.rsample()
            else:
                return Xold + gp_mean_out

    def rollout(self, num_steps, X0, X1=None, flg_sample_X=False, flg_sample_Y=False, flg_noise=False):
        """
        Generate a rollout of length 'num_step'. Return latent and observation trajectories
        
        Parameters
        ----------

        num_steps : int
            rollout length

        X0 : tensor(dtype)
            latent state at t=0

        X1 : tensor(dtype) (optionla)
            latent state at t=1

        flg_sample_X : boolean (optional)
            sample GP X output

        flg_sample_Y : boolean (optional)
            sample GP Y output

        flg_noise : boolean (optional)
            add noise to prediction variance

        Return
        ------
        X_hat : latent state rollout

        Y_hat : observation rollout

        """

        with torch.no_grad():
            X_hat = torch.zeros((num_steps, self.d), dtype=self.dtype, device=self.device)
            if X1 is None:
                X1 = X0

            # init latent variables
            X_hat[0,:] = torch.tensor(X0, dtype=self.dtype, device=self.device)
            t_start = 0
            if self.dyn_back_step == 2:
                X_hat[1,:] = torch.tensor(X1, dtype=self.dtype, device=self.device)
                t_start = 1

            # generate latent rollout
            for t in range(t_start,num_steps):
                if self.dyn_back_step == 1:
                    Xin = X_hat[t:t+1,:]
                if self.dyn_back_step == 2:
                    Xin = torch.cat((X_hat[t:t+1,:],X_hat[t-1:t,:]),1)
                mean_Xout_pred, var_Xout_pred = self.map_x_dynamics(Xin, flg_noise)
                X_hat[t+1:t+2,:] = self.get_next_x(mean_Xout_pred, var_Xout_pred, X_hat[t:t+1,:], flg_sample=flg_sample_X)

            # map to observation space
            mean_Y_pred, var_Y_pred = self.map_x_to_y(X_hat, flg_noise)
            if flg_sample_Y:
                distribution = Normal(mean_Y_pred, torch.sqrt(var_Y_pred)) 
                Y_hat = distribution.rsample()
            else:
                Y_hat = mean_Y_pred

            # return X_hat, Y_hat
            return X_hat.detach().cpu().numpy(), Y_hat.detach().cpu().numpy()


    def get_dynamics_map_performance(self, flg_noise=False):
        """
        Measure accuracy in latent dynamics prediction

        Parameters
        ---------- 

        flg_noise : boolean (optional)
            add noise to prediction variance
        
        Return
        ------

        mean_Xout_pred : mean of Xout prediction

        var_Xout_pred : variance of Xout prediction

        Xout : Xout matrix

        Xin : Xin matrix

        NMSE : Normalized Mean Square Error

        """

        with torch.no_grad():
            Xin, Xout, _ = self.get_Xin_Xout_matrices()
            mean_Xout_pred, var_Xout_pred = self.map_x_dynamics(Xin,flg_noise=flg_noise)

            mean_Xout_pred = mean_Xout_pred.clone().detach().cpu().numpy()
            var_Xout_pred = var_Xout_pred.clone().detach().cpu().numpy()
            Xout = Xout.clone().detach().cpu().numpy()
            Xin = Xin.clone().detach().cpu().numpy()

        return mean_Xout_pred, var_Xout_pred, Xout, Xin


    def get_latent_map_performance(self, flg_noise=False):
        """
        Measure accuracy of latent mapping

        Parameters
        ----------

        flg_noise : boolean (optional)
            add noise to prediction variance
        
        Return
        ------

        mean_Y_pred : mean of Y prediction

        var_Y_pred : variance of Y prediction

        Y : True observation matrix

        NMSE : Normalized Mean Square Error
        
        """

        with torch.no_grad():
            mean_Y_pred, var_Y_pred = self.map_x_to_y(self.X, flg_noise=flg_noise)

            mean_Y_pred = mean_Y_pred.clone().detach().cpu().numpy()
            var_Y_pred = var_Y_pred.clone().detach().cpu().numpy()

            Y = self.get_Y() + self.meanY

            return mean_Y_pred, var_Y_pred, Y

    def save(self, folder):
        """
        Save model

        Parameters
        ----------

        folder : string
            path of the desired save folder

        """

        print('\n### Save init data and model ###')
        torch.save(self.state_dict(), folder+'gpdm_state_dict.pt')
        config_dict={}
        config_dict['observations_list'] = self.observations_list
        config_dict['dyn_target'] = self.dyn_target
        config_dict['dyn_back_step'] = self.dyn_back_step
        config_dict['D'] = self.D
        config_dict['d'] = self.d
        config_dict['sigma_n_num_X'] = self.sigma_n_num_X
        config_dict['sigma_n_num_Y'] = self.sigma_n_num_Y
        pickle.dump(config_dict, open(folder+'gpdm_config_dict.pt', 'wb'))


    def load(self, config_dict, state_dict, flg_print = False):
        """
        Load (previously initialized) model

        Parameters
        ----------

        config_dict : dict
            configuration dictionary

        config_dict : collections.OrderedDict
            model state dictionary

        flg_print : bool (optional)
            flag to print loaded state_dict (default is False)
        """

        self.observations_list = config_dict['observations_list']
        self.init_X()
        self.load_state_dict(state_dict)

        # save inverse kernel matrices after training
        Ky = self.get_y_kernel(self.X,self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv,U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin,Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv,U_inv.t())

        if flg_print:
            print("Loaded model's state_dict:")
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor])


class CGPDM(GPDM):
    """
    Controlled Gaussian Process Dynamical Model

    Additional Attributes
    ---------------------

    u_dim : int
        control input dimension

    """
    def __init__(self, D, d, u_dim, dyn_target, dyn_back_step,
                 y_lambdas_init, y_lengthscales_init, y_sigma_n_init,
                 x_lambdas_init, x_lengthscales_init, x_sigma_n_init, x_lin_coeff_init,
                 flg_train_y_lambdas=True, flg_train_y_lengthscales=True, flg_train_y_sigma_n=True,
                 flg_train_x_lambdas=True, flg_train_x_lengthscales=True, flg_train_x_sigma_n=True, flg_train_x_lin_coeff=True,
                 sigma_n_num_Y=0., sigma_n_num_X=0., dtype=torch.float64, device=torch.device('cpu')):
        """
        Additional Parameters
        ---------------------
        
        u_dim : int
            control input dimension

        controls_list
            list of observed control actions

        """
        super(CGPDM,self).__init__(D=D, d=d, dyn_target=dyn_target, dyn_back_step=dyn_back_step,
                 y_lambdas_init=y_lambdas_init, y_lengthscales_init=y_lengthscales_init, y_sigma_n_init=y_sigma_n_init,
                 x_lambdas_init=x_lambdas_init, x_lengthscales_init=x_lengthscales_init, x_sigma_n_init=x_sigma_n_init, x_lin_coeff_init=x_lin_coeff_init,
                 flg_train_y_lambdas=flg_train_y_lambdas, flg_train_y_lengthscales=flg_train_y_lengthscales, flg_train_y_sigma_n=flg_train_y_sigma_n,
                 flg_train_x_lambdas=flg_train_x_lambdas, flg_train_x_lengthscales=flg_train_x_lengthscales,
                 flg_train_x_sigma_n=flg_train_x_sigma_n, flg_train_x_lin_coeff=flg_train_x_lin_coeff,
                 sigma_n_num_Y=sigma_n_num_Y, sigma_n_num_X=sigma_n_num_X, dtype=dtype, device=device)

        # control input dimension
        self.u_dim = u_dim
    
        self.controls_list = []


    def add_data(self, Y, U):
        """
        Add observation  and control data to self.observations_list

        Parameters
        ----------

        Y : double
            observation data (dimension: N x D)

        U : double
            observation data (dimension: N x u_dim)

        """
        if Y.shape[1]!=self.D:
            raise ValueError('Y must be a N x D matrix collecting observation data!')
        if U.shape[1]!=self.u_dim:
            raise ValueError('U must be a N x u_dim matrix collecting observation data!')
        if U.shape[0]!=Y.shape[0]:
            raise ValueError('Y and U must have the same number N of data!')

        self.observations_list.append(Y)
        self.controls_list.append(U)

        self.num_sequences = self.num_sequences+1

        print('Num. of sequences = '+str(self.num_sequences)+' [Total data points = '+str(np.concatenate(self.observations_list, 0).shape[0])+']')

    def get_Xin_Xout_matrices(self, U=None, X=None, target=None, back_step=None):
        """
        Compute GP input and output matrices (Xin, Xout) for GP X

        Parameters
        ----------

        U : tensor(dtype) (optional)
            control input matrix

        X : tensor(dtype) (optional)
            latent state matrix

        target : string (optional)
            dynamic map function target ('full' or 'delta')
        
        back_step : int (optional)
            memory of dynamic map function (1 or 2)
        
        Return
        ------
        Xin : GP X input matrix

        Xout : GP X output matrix

        start_indeces : list of sequences' start indeces

        """

        if X==None:
            X = self.X
        if U==None:
            U = self.controls_list
        if target==None:
            target = self.dyn_target
        if back_step==None:
            back_step = self.dyn_back_step

        X_list = []
        U_list = []
        x_start_index = 0
        start_indeces = []
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(X[x_start_index:x_start_index+sequence_length,:])
            U_list.append(torch.tensor(U[j], dtype=self.dtype, device=self.device))
            start_indeces.append(x_start_index)
            x_start_index = x_start_index+sequence_length           

        if target == 'full' and back_step == 1:
            # in: x(t)
            Xin = torch.cat((X_list[0][0:-1,:],U_list[0][0:-1,:]), 1)
            # out: x(t+1)
            Xout = X_list[0][1:,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][0:-1,:],U_list[j][0:-1,:]), 1)),0)
                Xout = torch.cat((Xout, X_list[j][1:,:]),0)

        elif target == 'full' and back_step == 2:
            # in: [x(t), x(t-1)]
            Xin = torch.cat((X_list[0][1:-1,:],X_list[0][0:-2,:],U_list[0][1:-1,:],U_list[0][0:-2,:]), 1)
            # out: x(t+1)
            Xout = X_list[0][2:,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][1:-1,:], X_list[j][0:-2,:], U_list[j][1:-1,:], U_list[j][0:-2,:]), 1)),0)
                Xout = torch.cat((Xout, X_list[j][2:,:]),0)

        elif target == 'delta' and back_step == 1:
            # in: x(t)
            Xin = torch.cat((X_list[0][0:-1,:],U_list[0][0:-1,:]), 1)
            # out: x(t+1)-x(t)
            Xout = X_list[0][1:,:] - X_list[0][0:-1,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][0:-1,:],U_list[j][0:-1,:]), 1)),0)
                Xout = torch.cat((Xout, X_list[j][1:,:] - X_list[j][0:-1,:]),0)

        elif target == 'delta' and back_step == 2:
            # in: [x(t), x(t-1)]
            Xin = torch.cat((X_list[0][1:-1,:], X_list[0][0:-2,:], U_list[0][1:-1,:], U_list[0][0:-2,:]), 1)
            # out: x(t+1)-x(t)
            Xout = X_list[0][2:,:] - X_list[0][1:-1,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][1:-1,:], X_list[j][0:-2,:], U_list[j][1:-1,:], U_list[j][0:-2,:]), 1)),0)
                Xout = torch.cat((Xout, X_list[j][2:,:] - X_list[j][1:-1,:]),0)
        
        else:
            raise ValueError('target must be either full or delta \n back_step must be either 1 or 2')

        return Xin, Xout, start_indeces



    def rollout(self, num_steps, control_actions, X0, X1=None, flg_sample_X=False, flg_sample_Y=False, flg_noise=False):
        """
        Generate a rollout of length 'num_steps'. Return latent and observation trajectories
        
        Parameters
        ----------

        num_steps : int
            rollout length
        
        control_actions : tensor(dtype) 
            list of control inputs (dimension: N x u_dim)
        X0 : tensor(dtype)
            latent state at t=0

        X1 : tensor(dtype) (optionla)
            latent state at t=1

        flg_sample_X : boolean (optional)
            sample GP X output

        flg_sample_Y : boolean (optional)
            sample GP Y output

        flg_noise : boolean (optional)
            add noise to prediction variance

        Return
        ------
        X_hat : latent state rollout

        Y_hat : observation rollout

        """

        if control_actions.shape[0]!=num_steps:
            raise ValueError('len(control_actions) must be equal to num_steps!')

        with torch.no_grad():
            X_hat = torch.zeros((num_steps, self.d), dtype=self.dtype, device=self.device)
            if X1 is None:
                X1 = X0

            # init latent variables
            X_hat[0,:] = torch.tensor(X0, dtype=self.dtype, device=self.device)
            t_start = 0
            if self.dyn_back_step == 2:
                X_hat[1,:] = torch.tensor(X1, dtype=self.dtype, device=self.device)
                t_start = 1

            # generate latent rollout
            for t in range(t_start,num_steps):
                if self.dyn_back_step == 1:
                    Xin = torch.cat((X_hat[t:t+1,:], control_actions[t:t+1,:]),1)
                if self.dyn_back_step == 2:
                    Xin = torch.cat((X_hat[t:t+1,:], X_hat[t-1:t,:], control_actions[t:t+1,:], control_actions[t-1:t,:]),1)
                mean_Xout_pred, var_Xout_pred = self.map_x_dynamics(Xin, flg_noise)
                X_hat[t+1:t+2,:] = self.get_next_x(mean_Xout_pred, var_Xout_pred, X_hat[t:t+1,:], flg_sample=flg_sample_X)

            # map to observation space
            mean_Y_pred, var_Y_pred = self.map_x_to_y(X_hat, flg_noise)
            if flg_sample_Y:
                distribution = Normal(mean_Y_pred, torch.sqrt(var_Y_pred)) 
                Y_hat = distribution.rsample()
            else:
                Y_hat = mean_Y_pred

            # return X_hat, Y_hat
            return X_hat.detach().cpu().numpy(), Y_hat.detach().cpu().numpy()


    def save(self, folder):
        """
        Save model

        Parameters
        ----------

        folder : string
            path of the desired save folder

        """

        print('\n### Save init data and model ###')
        torch.save(self.state_dict(), folder+'cgpdm_state_dict.pt')
        config_dict={}
        config_dict['observations_list'] = self.observations_list
        config_dict['controls_list'] = self.controls_list
        config_dict['dyn_target'] = self.dyn_target
        config_dict['dyn_back_step'] = self.dyn_back_step
        config_dict['D'] = self.D
        config_dict['d'] = self.d
        config_dict['u_dim'] = self.u_dim
        config_dict['sigma_n_num_X'] = self.sigma_n_num_X
        config_dict['sigma_n_num_Y'] = self.sigma_n_num_Y
        pickle.dump(config_dict, open(folder+'cgpdm_config_dict.pt', 'wb'))


    def load(self, config_dict, state_dict, flg_print = False):
        """
        Load (previously initialized) model

        Parameters
        ----------

        config_dict : dict
            configuration dictionary

        config_dict : collections.OrderedDict
            model state dictionary

        flg_print : bool (optional)
            flag to print loaded state_dict (default is False)
        """

        self.observations_list = config_dict['observations_list']
        self.controls_list = config_dict['controls_list']
        self.init_X()
        self.load_state_dict(state_dict)

        # save inverse kernel matrices after training
        Ky = self.get_y_kernel(self.X,self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv,U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin,Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv,U_inv.t())

        if flg_print:
            print("Loaded model's state_dict:")
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor])