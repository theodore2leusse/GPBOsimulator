# Theodore 
# first GPBO simu run
# the code is modified from Rose's work

## ------------------------ IMPORT LIB ------------------------ ##

import math
import torch
import gpytorch
import botorch

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from scipy.stats import multivariate_normal
from scipy.stats import norm

from botorch.acquisition.analytic import (UpperConfidenceBound, ExpectedImprovement,
                                          NoisyExpectedImprovement, LogExpectedImprovement,
                                          LogNoisyExpectedImprovement)
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from utils import *
from scipy.stats import multivariate_normal
import pandas as pd
import warnings
import time

## ------------------------ IMPORT FILES ------------------------ ##

from DataSet import DataSet

## ------------------------ OPTI ------------------------ ##

if __name__ == "__main__":

    warnings.filterwarnings("ignore") # supress /usr/local/lib/python3.10/dist-packages/botorch/models/utils/assorted.py:173: InputDataWarning: Input data is not
    #contained to the unit cube. Please consider min-max scaling the input data. warnings.warn(msg, InputDataWarning) at each iteration

    np.random.seed(1) # allow to repeat the same 'random' simu
    torch.manual_seed(1) # allow to repeat the same 'random' simu

    ## ------------------------ DEFINE CONSTANTS ------------------------ ##

    NB_REP = 1 # number of repetition
    NB_RND = 1 # number of times we will randomly select the next query
               # before using the acquisition function.     
    KAPPA = 10 # parameter for the acquisition function 

    AF = 'EI'  #  choice of the acquisition fonction among 'EI', 'NEI', 'logEI', 'logNEI'

    ## ------------------------ Data Set ------------------------ ##

    ds = DataSet('data/','nhp','Cebus1_M1_190221.mat','first_GPBO_realisticResponses')
    ds.load_matlab_data() # load data from the dataset_file
        
    nb_emg = len(ds.set['emgs'])          # nb of emgs
    space_size = ds.set['ch2xy'].shape[0] # nb of electrodes in our chip
    space_dim = ds.set['ch2xy'].shape[1]  # nb of dimensions in our entry space
    nb_it = space_size                    # nb of iteration in the BO
    X_test = ds.set['ch2xy'].astype('float')
    X_test_normed = torch.from_numpy((ds.set['ch2xy'] - np.min(ds.set['ch2xy'], axis = 0 ))/
                                     (np.max(ds.set['ch2xy'], axis = 0 ) - 
                                      np.min(ds.set['ch2xy'], axis = 0))) # normalized coordonates in [0,1]
    X_test_normed = X_test_normed.double()   

    # identify the node in the border of our hypercube [0,1]²
    border_idx = np.union1d(np.where((X_test_normed ==1))[0], np.where((X_test_normed ==0))[0])
    
    # create always the same tensor (if the seed is specified before)
    # rand_idx tensor will give for each combination of emg & repetition space_size choose NB_RND 
    rand_idx = np.zeros((nb_emg, NB_REP, NB_RND), dtype=int)
    for i in range(nb_emg):
        for j in range(NB_REP):
            rand_idx[i, j,:] = np.random.permutation(np.arange(space_size))[:NB_RND]
            
    # initialization of several tensors to stock the data 
    P_test_x = torch.zeros((nb_emg, NB_REP, space_dim, nb_it), 
                           dtype=torch.float64)               # list queries' entry coordinates
    P_test_x_idx = torch.zeros((nb_emg, NB_REP, 1, nb_it),
                                dtype=int)                    # list queries' entry id
    P_test_y = torch.zeros((nb_emg, NB_REP, 1, nb_it),
                            dtype=torch.float64)              # list queries' responses 
    best_pred_x = torch.zeros((nb_emg, NB_REP, 1, nb_it),
                               dtype=int)                     # list the electrode_id of the best prediction 
    best_pred_x_measured = torch.zeros((nb_emg, NB_REP, 1, nb_it),
                               dtype=int)                     # list the electrode_id of the best prediction 
                                                              # including only electrodes that have already be measured
    P_mean_pred = torch.zeros((nb_emg, NB_REP, 1, nb_it, space_size), # save the mean distribution pred for each it
                         dtype=torch.float64)
    P_std_pred = torch.zeros((nb_emg, NB_REP, 1, nb_it, space_size), # save the std distribution pred for each it
                         dtype=torch.float64)

    tic = time.time()

    # loop over the different emgs 
    for emg_i in range(nb_emg):
        
        print('emg', emg_i)
        
        # loop over the different repetitions 
        for r in range(NB_REP):     
            
            queried_loc = [] # initialization of the list of the next_x_idx

            # loop over the number of node in our grid
            for i in range(nb_it):

                if i < NB_RND: # we chose the next node randomly with next_x_idx

                    next_x_idx = rand_idx[emg_i, r, i]
                    
                else: # we have to choose an acquisition function 
                    
                    if AF == 'EI':      
                        mu_sample_opt = torch.max(train_Y) # max of the queries' responses 
                        AF_ = ExpectedImprovement(gp, best_f = mu_sample_opt, maximize = True)

                    elif AF == 'NEI':                       
                        AF_ = NoisyExpectedImprovement(gp, train_X, num_fantasies=20, maximize = True) 
                        
                    elif AF == 'logEI':                       
                        mu_sample_opt = torch.max(train_Y) # max of the queries' responses
                        AF_ = LogExpectedImprovement(gp, best_f = mu_sample_opt, maximize = True)                      
                        
                    elif AF == 'logNEI':                        
                        AF_ = LogNoisyExpectedImprovement(gp, train_X, num_fantasies=20, maximize = True)  
                                               
                    elif AF == 'UCB':                  
                        AF_ = UpperConfidenceBound(gp, beta = KAPPA, maximize = True)
                                       
                    ei_val = AF_(X_test_normed[:, None, :]) # the arg has to be in the shape 
                                                            # (space_size,1,space_dim)
                                                            # ei_val is a tensor of size space_size
                    af_val = ei_val.detach().numpy()        # af_val is an array of size space_size  

                    # choice of the next location we will try our black box function
                    if next_x_idx.size > 1:    
                        next_x_idx = np.random.choice(np.where(af_val==af_val.max())[0])
                    else: 
                        next_x_idx = np.where(af_val==af_val.max())[0][0]
                
                queried_loc.append(next_x_idx) # update the list with the next_x_idx we have just chosen
                next_x = X_test_normed[next_x_idx] # entry coordonates for the next query
                resp = ds.get_realistic_response(emg_i, next_x_idx) # get the response of this query (float)

                P_test_x[emg_i, r, :, i] = next_x             # update the tensor 
                P_test_x_idx[emg_i, r, 0, i] = next_x_idx     # update the tensor
                P_test_y[emg_i, r, 0, i] = resp.astype(float) # update the tensor

                train_X = P_test_x[emg_i, r, :, :int(i+1)] # only focus on what have been updated in
                                                           # P_test_x[emg_i, r] (2d-tensor)
                train_X = train_X.T                        # transpose the tensor => shape(2,nb_queries)

                train_Y = P_test_y[emg_i, r, 0, :int(i+1)] # only focus on what have been updated in
                                                           # P_test_y[emg_i, r] (1d-tensor)
                train_Y = train_Y[...,np.newaxis]          # add a new final dimension in the tensor 
                                                           # (2d-tensor) train_y is a column matrix

                if i == 0:
                    if resp != 0:
                        train_Y = train_Y/train_Y.max() # =1 
                else:
                    train_Y = standardize(train_Y) # (data-mean)/std

                if AF == 'NEI' or AF == 'logNEI':   
                    train_Yvar = torch.full_like(train_Y, 0.15) # create a tensor which has the same shape
                                                                # and type as train_y, but is filled with 
                                                                # a constant value of 0.15.
                                                                # train_Yvar allows to indicate the observed 
                                                                # measurement noise for each measure
                    gp = SingleTaskGP(train_X, train_Y, train_Yvar = train_Yvar) # create the GP model
                    
                else:
                    gp = SingleTaskGP(train_X, train_Y) # create the GP model
                
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # create the log likelihood function 
                fit_gpytorch_mll(mll) # optimize the hyperparameters of the GP 
                gp_mean_pred = gp.posterior(X_test_normed).mean.flatten() # GP prediction for the mean
                best_pred_x[emg_i, r, 0, i] = torch.argmax(gp_mean_pred).item() # update the tensor
                
                gp_std_pred = torch.sqrt(gp.posterior(X_test_normed).variance.flatten())

                P_mean_pred[emg_i, r, 0, i, :] = gp_mean_pred   # update the tensor 
                P_std_pred[emg_i, r, 0, i, :] = gp_std_pred     # update the tensor

                # sets to -inf the means of the nodes that have not already been measured
                not_queried_loc = np.setdiff1d(np.arange(space_size), queried_loc)
                gp_mean_pred[not_queried_loc] = -np.inf

                best_pred_x_measured[emg_i, r, 0, i] = torch.argmax(gp_mean_pred).item() # update the tensor
        

    tac = time.time()

    # Calculate and display the elapsed time
    elapsed_time = tac - tic
    print(f"Elapsed time: {elapsed_time} seconds")

    np.savez('results/gpbo_{0}_{1}'.format(AF, ds.dataset_name), 
             P_test_x = P_test_x,
             P_test_x_idx = P_test_x_idx,
             P_test_y = P_test_y, 
             P_mean_pred = P_mean_pred.detach().numpy(),
             P_std_pred = P_std_pred.detach().numpy(),
             best_pred_x = best_pred_x,
             best_pred_x_measured = best_pred_x_measured,
             rand_idx = rand_idx,
             elapsed_time = elapsed_time,
             **ds.set)
