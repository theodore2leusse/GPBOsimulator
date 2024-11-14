# Theodore 
# first GPBO simu run
# the code is modified from Rose's work

## ------------------------ IMPORT LIB ------------------------ ##

# import math
# import torch
# import gpytorch
# import botorch

# import matplotlib.pyplot as plt

# import numpy as np
# from scipy.stats import norm

# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_mll
# from botorch.utils.transforms import standardize, normalize
# from gpytorch.mlls import ExactMarginalLogLikelihood

# from scipy.stats import multivariate_normal
# from scipy.stats import norm

# from botorch.acquisition.analytic import (UpperConfidenceBound, ExpectedImprovement,
#                                           NoisyExpectedImprovement, LogExpectedImprovement,
#                                           LogNoisyExpectedImprovement)
# from botorch.optim import optimize_acqf, optimize_acqf_discrete

# from utils import *
# from scipy.stats import multivariate_normal
# import pandas as pd
# import warnings
# import time
# from tqdm import tqdm

## ------------------------ IMPORT FILES ------------------------ ##

from DataSet import DataSet
from SimGPBO import SimGPBO

if __name__ == "__main__":

    ds = DataSet('data/','nhp','Cebus1_M1_190221.mat','first_GPBO_validResponses')
    ds.load_matlab_data() # load data from the dataset_file

    sim_gpbo = SimGPBO(name = '1st_try', 
                       ds = ds,
                       AF = 'EI',
                       NB_REP = 2,
                       NB_IT = 60,
                       )
    
    # sim_gpbo.select_emgs([0,1])

    sim_gpbo.set_custom_gp_hyperparameters(noise_std= 0.4, lengthscale= 0.01)

    sim_gpbo.run_simulations(gp_origin = 'custom_FixedOnlineGP', mean_and_std_storage = True)