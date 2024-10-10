# sciNeurotech Lab 
# Theodore
"""
in this file, we will define a class in order simulate GPBO 
"""

# import lib
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import (UpperConfidenceBound, 
                                          ExpectedImprovement,
                                          NoisyExpectedImprovement, 
                                          LogExpectedImprovement,
                                          LogNoisyExpectedImprovement)
import numpy as np
from utils import *
import warnings
import time
from tqdm import tqdm

# import file 
from DataSet import DataSet


class SimGPBO():
    """
    Class to simulate GPBO

    Attributes:
        ???
    
    Methods: 
        ???
    """

    def __init__(self, name: str, ds: DataSet, AF: str, NB_REP: int = 1, NB_RND: int = 1, NB_IT: int = None,
                 KAPPA: float | torch.Tensor = None) -> None:
        """
        initialize a SimGPBO instance

        Args:
            ds (DataSet): dataset from the DataSet class
            AF (str): Acquisition Function for each BO. 
                      choice of the acquisition fonction among
                      'EI', 'NEI', 'logEI', 'logNEI', 'UCB'
            NB_REP (int, optional): nb of repetitions. Defaults to 1.
            NB_RND (int, optional): number of times we will randomly select the next 
                                    query before using the acquisition function. 
                                    Defaults to 1.
            NB_IT (int, optional): nb of iterations for each BO. Defaults to None.
            KAPPA (float | torch.Tensor, optional): KAPPA parameter for the UCB AF.
                                                    Defaults to None.
        """
        # we need to load data from our dataset, if this isn't already the case. 
        if len(ds.set)  == 0:
            ds.load_matlab_data()
        if AF == 'UCB' & KAPPA == None:
            raise ValueError("KAPPA is not specified while UCB is the chosen acquisition function")

        self.name = name
        self.ds = ds
        self.AF = AF                     
        self.NB_REP = NB_REP
        self.NB_RND = NB_RND
        self.KAPPA = KAPPA

        self.nb_emg = len(self.ds.set['emgs'])          # nb of emgs
        self.space_size = self.ds.set['ch2xy'].shape[0] # nb of electrodes in our chip
        self.space_dim = self.ds.set['ch2xy'].shape[1]  # nb of dimensions in our entry space

        if NB_IT is None: # if NB_IT not specified 
            self.NB_IT = self.space_size   
        else:
            self.NB_IT = NB_IT
        
    def select_emgs(self) -> None: # l'idée est pouvoir sélectionner les emgs que tu veux simuler 
        # modification de self.ds 
        self.nb_emg = len(self.ds.set['emgs'])

    def save_to_npz_file(self):# probablement pas de besoin si on automatise l'enregistrement dans run_simulations
        pass

    def normalized_entry_space(self) -> torch.Tensor:
        """TODO

        Returns:
            torch.Tensor: _description_
        """
        X_test_normed = torch.from_numpy((self.ds.set['ch2xy'] - np.min(self.ds.set['ch2xy'], axis = 0 ))/
                                            (np.max(self.ds.set['ch2xy'], axis = 0 ) - 
                                            np.min(self.ds.set['ch2xy'], axis = 0))) # normalized coordonates in [0,1]
        return(X_test_normed.double())
    
    def get_rand_idx(self) -> np.array:
        """TODO

        Returns:
            np.array: _description_
        """
        rand_idx = np.zeros((self.nb_emg, self.NB_REP, self.NB_RND), dtype=int)
        for i in range(self.nb_emg):
            for j in range(self.NB_REP):
                rand_idx[i, j,:] = np.random.permutation(np.arange(self.space_size))[:self.NB_RND]
        return(rand_idx)
    
    def initialize_storage_basic_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """TODO

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        # initialization of several tensors to stock the data 
        P_test_x = torch.zeros((self.nb_emg, self.NB_REP, self.space_dim, self.NB_IT), 
                            dtype=torch.float64)            # list queries' entry coordinates
        P_test_x_idx = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
                                    dtype=int)              # list queries' entry id
        P_test_y = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
                                dtype=torch.float64)        # list queries' responses 
        best_pred_x = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
                                dtype=int)                  # list the electrode_id of the best prediction 
        best_pred_x_measured = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
                                dtype=int)                  # list the electrode_id of the best prediction 
                                                            # including only electrodes that have already be measured
        return(P_test_x, P_test_x_idx, P_test_y, best_pred_x, best_pred_x_measured)

    def initialize_storage_clock_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """TODO

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        iter_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),      
                            dtype=torch.float64) # save the iteration duration for each it
        hyp_opti_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),  
                            dtype=torch.float64) # save the hyperparameter optimization duration for each it
        mean_calc_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT), 
                            dtype=torch.float64) # save the mean calculation duration for each it
        std_calc_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT), 
                            dtype=torch.float64) # save the std calculation duration for each it
        return(iter_durations, hyp_opti_durations, mean_calc_durations, std_calc_durations)

    def run_simulations(self, npz_save: str = 'end', manual_seed: bool = True, filterwarning: bool = True, mean_and_std_storage: bool = False) -> None:
        
        if filterwarning:
            warnings.filterwarnings("ignore") # supress /usr/local/lib/python3.10/dist-packages/botorch/models/utils/assorted.py:173: InputDataWarning: Input data is not
                                              # contained to the unit cube. Please consider min-max scaling the input data. warnings.warn(msg, InputDataWarning) at each iteration
        if manual_seed:
            np.random.seed(1) # allow to repeat the same 'random' simu
            torch.manual_seed(1) # allow to repeat the same 'random' simu 

        # we normalize the entry space
        X_test_normed = self.normalized_entry_space()
        # X_test = self.ds.set['ch2xy'].astype('float')
        # X_test_normed = torch.from_numpy((self.ds.set['ch2xy'] - np.min(self.ds.set['ch2xy'], axis = 0 ))/
        #                                     (np.max(self.ds.set['ch2xy'], axis = 0 ) - 
        #                                     np.min(self.ds.set['ch2xy'], axis = 0))) # normalized coordonates in [0,1]
        # X_test_normed = X_test_normed.double()  

        # create always the same tensor (if the seed is specified before)
        # rand_idx tensor will give for each combination of emg & repetition space_size choose NB_RND 
        rand_idx = self.get_rand_idx()
        # rand_idx = np.zeros((self.nb_emg, self.NB_REP, self.NB_RND), dtype=int)
        # for i in range(self.nb_emg):
        #     for j in range(self.NB_REP):
        #         rand_idx[i, j,:] = np.random.permutation(np.arange(self.space_size))[:self.NB_RND]

        # initialization of several tensors to stock the data 
        P_test_x, P_test_x_idx, P_test_y, best_pred_x, best_pred_x_measured = self.initialize_storage_basic_tensors()
        # P_test_x = torch.zeros((self.nb_emg, self.NB_REP, self.space_dim, self.NB_IT), 
        #                     dtype=torch.float64)            # list queries' entry coordinates
        # P_test_x_idx = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
        #                             dtype=int)              # list queries' entry id
        # P_test_y = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
        #                         dtype=torch.float64)        # list queries' responses 
        # best_pred_x = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
        #                         dtype=int)                  # list the electrode_id of the best prediction 
        # best_pred_x_measured = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),
        #                         dtype=int)                  # list the electrode_id of the best prediction 
        #                                                     # including only electrodes that have already be measured
        iter_durations, hyp_opti_durations, mean_calc_durations, std_calc_durations = self.initialize_storage_clock_tensors()
        # iter_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),      
        #                     dtype=torch.float64) # save the iteration duration for each it
        # hyp_opti_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT),  
        #                     dtype=torch.float64) # save the hyperparameter optimization duration for each it
        # mean_calc_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT), 
        #                     dtype=torch.float64) # save the mean calculation duration for each it
        # std_calc_durations = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT), 
        #                     dtype=torch.float64) # save the std calculation duration for each it
        if mean_and_std_storage:
            P_mean_pred = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT, self.space_size), 
                                dtype=torch.float64) # save the mean distribution pred for each it
            P_std_pred = torch.zeros((self.nb_emg, self.NB_REP, 1, self.NB_IT, self.space_size), 
                                dtype=torch.float64) # save the std distribution pred for each it




