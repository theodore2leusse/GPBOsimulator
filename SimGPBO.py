# sciNeurotech Lab 
# Theodore
"""
in this file, we will define a class in order simulate GPBO 
"""

# import lib
import torch
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
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
    Class to simulate Gaussian Process Bayesian Optimization (GPBO) experiments.

    This class provides methods to run simulations for GPBO over a defined number of
    electro-myographic (EMG) signals and repetitions. It allows for both random and
    acquisition-based querying strategies, determined by the number of iterations.

    Attributes:
        name (str): The name of the simulation.
        ds (DataSet): The dataset from the DataSet class.
        AF (str): The Acquisition Function for each BO. Choices include 'EI', 'NEI', 'logEI', 'logNEI', and 'UCB'.
        NB_REP (int): The number of repetitions. Defaults to 1.
        NB_RND (int): The number of times a random selection is made before using the acquisition function. Defaults to 1.
        NB_IT (int): The number of iterations for each BO. Defaults to None.
        KAPPA (float | torch.Tensor): The KAPPA parameter for the UCB AF. Defaults to None.
        nb_emg (int): The number of EMG signals in the dataset.
        space_size (int): The size of the electrodes in the chip.
        space_dim (int): The number of dimensions in the entry space.
        num_fantasies (int): The number of fantasies for the acquisition function. Defaults to 20.

    Methods:
        select_emgs(emg_idx: list[int]) -> None:
            Selects specific electromyography (EMG) signals for simulation.

        normalized_entry_space() -> torch.Tensor:
            Normalizes the entry space coordinates to a [0, 1] range.

        get_rand_idx() -> np.ndarray:
            Generates random indices for each EMG channel and repetition, selecting a subset of the search space.

        initialize_storage_basic_tensors() -> None:
            Initializes the basic tensors used to store key data during the optimization process.

        initialize_storage_clock_tensors() -> None:
            Initializes the tensors used to store timing information for various stages of the optimization process.

        initialize_storage_mean_and_std_tensors() -> None:
            Initializes the tensors used to store the predicted mean and standard deviation for each iteration.

        select_next_query(AF: str, gp, train_X: torch.Tensor, train_Y: torch.Tensor, X_test_normed: torch.Tensor) -> int:
            Selects the next query point based on the acquisition function (AF) applied to the normalized test points.

        get_response(emg_i: int, next_x_idx: int, response_type: str = 'valid') -> float:
            Retrieves the response for a given query based on the specified response type.

        npz_save(clock_storage: bool, mean_and_std_storage: bool, final: bool = False, emg_i: int = None, r: int = None) -> None:
            Saves the GPBO results to a .npz file, with optional storage of mean/std predictions and clock durations.

        erase_storage() -> None:
            Erases stored attributes to free up memory after saving results.

        run_simulations(manual_seed: bool = True, clock_storage: bool = True, mean_and_std_storage: bool = False, intermediate_save: bool = False, response_type: str = 'valid') -> None:
            Execute multiple simulations for Gaussian Process Bayesian Optimization (GPBO) over a defined number of EMG signals and repetitions.
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
        if AF == 'UCB' and self.KAPPA is None:
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

        # initialisation default values 
        self.num_fantasies = 20
        
    def select_emgs(self, emg_idx: list[int]) -> None: 
        """
        Selects specific electromyography (EMG) signals for simulation.

        This method modifies the dataset by selecting the EMG signals specified 
        by the indices provided in `emg_idx`. It updates the relevant fields in 
        the dataset to include only the selected EMG signals and their corresponding 
        responses and validity indicators.

        Args:
            emg_idx (list[int]): A list of indices corresponding to the EMG signals 
                that should be selected for simulation.

        Returns:
            None: This method does not return a value. It modifies the dataset 
                in place.

        Example:
            >>> obj.select_emgs([0, 2, 3])
            This will select the EMG signals at indices 0, 2, and 3.
        """
        # modification de self.ds 
        self.ds.set['emgs'] = self.ds.set['emgs'][emg_idx]
        self.ds.set['sorted_resp'] = self.ds.set['sorted_resp'][:, emg_idx]
        self.ds.set['sorted_isvalid'] = self.ds.set['sorted_isvalid'][:, emg_idx]
        self.ds.set['sorted_respMean'] = self.ds.set['sorted_respMean'][:, emg_idx]
        self.nb_emg = len(self.ds.set['emgs'])

    def normalized_entry_space(self) -> torch.Tensor:
        """
        Normalizes the entry space coordinates to a [0, 1] range.

        This method normalizes the coordinates of the entries stored in `self.ds.set['ch2xy']` by subtracting the 
        minimum value and dividing by the range (max - min) for each dimension. The result is a tensor where all 
        coordinates are scaled to fall within the range [0, 1] for each dimension.

        Returns:
            torch.Tensor: A tensor of normalized coordinates (of type `torch.float64`) representing the input space 
                        for the Gaussian Process. The shape of the tensor is the same as `self.ds.set['ch2xy']`.
        
        Tensors:
            - `X_test_normed`: A PyTorch tensor containing the normalized coordinates of the test points. The normalization 
                            ensures that all coordinates are in the range [0, 1].
        """
        # Normalize the coordinates to the range [0, 1]
        X_test_normed = torch.from_numpy((self.ds.set['ch2xy'] - np.min(self.ds.set['ch2xy'], axis=0)) /
                                        (np.max(self.ds.set['ch2xy'], axis=0) - np.min(self.ds.set['ch2xy'], axis=0)))
        
        # Return the normalized tensor, converting to double precision (float64)
        return X_test_normed.double()
    
    def get_rand_idx(self) -> np.ndarray:
        """
        Generates random indices for each EMG channel and repetition, selecting a subset of the search space.

        This method creates a 3D array of randomly permuted indices, where the size of the third dimension corresponds
        to the number of random selections (NB_RND) from the search space (space_size). For each EMG channel (nb_emg) 
        and each repetition (NB_REP), a unique random permutation of indices is generated, and the first NB_RND 
        indices are selected.

        Returns:
            np.ndarray: A 3D array of shape (nb_emg, NB_REP, NB_RND) containing the randomly selected indices for each 
                        channel and repetition. These indices correspond to random selections from the search space.
        
        Tensors:
            - `rand_idx`: A 3D numpy array with dimensions `(nb_emg, NB_REP, NB_RND)`. Each entry corresponds to a 
                        randomly selected index from the space of size `space_size`.

        """
        # Initialize a tensor to store the random indices
        rand_idx = np.zeros((self.nb_emg, self.NB_REP, self.NB_RND), dtype=int)
        
        # Loop through each EMG channel and repetition to generate a random permutation of indices
        for i in range(self.nb_emg):
            for j in range(self.NB_REP):
                # Generate a random permutation of indices from the search space and select the first NB_RND
                rand_idx[i, j, :] = np.random.permutation(np.arange(self.space_size))[:self.NB_RND]
        
        # Return the 3D array of random indices
        return rand_idx

    def initialize_storage_basic_tensors(self) -> None:
        """
        Initializes the basic tensors used to store key data during the optimization process.

        This method sets up several tensors that will hold data related to the queries and their results during the 
        Bayesian optimization process. These tensors track information such as the coordinates of the queries, their 
        associated responses, and the indices of the best predictions.

        Attributes Initialized:
            - `P_test_x`: Stores the coordinates of the test queries.
            - `P_test_x_idx`: Stores the indices of the test queries.
            - `P_test_y`: Stores the responses associated with the test queries.
            - `best_pred_x`: Stores the indices of the best predicted points (electrode IDs).
            - `best_pred_x_measured`: Stores the indices of the best predicted points among already measured electrodes.

        Shapes:
            - All tensors are of shape `(nb_emg, NB_REP, ..., NB_IT)`.
            - `nb_emg`: Number of EMG channels.
            - `NB_REP`: Number of repetitions per test.
            - `space_dim`: Dimensionality of the search space.
            - `NB_IT`: Number of iterations.

        """
        # Initialize tensor to store the coordinates of test queries (space_dim dimensions for the coordinates)
        self.P_test_x = torch.zeros(
            (self.nb_emg, self.NB_REP, self.space_dim, self.NB_IT), 
            dtype=torch.float64
        )
        
        # Initialize tensor to store the indices of the test queries
        self.P_test_x_idx = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=int
        )
        
        # Initialize tensor to store the responses associated with the test queries
        self.P_test_y = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=torch.float64
        )
        
        # Initialize tensor to store the indices of the best predicted test points (electrode IDs)
        self.best_pred_x = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=int
        )
        
        # Initialize tensor to store the indices of the best predicted test points among already measured electrodes
        self.best_pred_x_measured = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=int
        )

    def initialize_storage_clock_tensors(self) -> None:
        """Initializes the tensors used to store timing information for various stages of the optimization process.

        This method sets up four tensors to track the duration of key operations during each iteration of the optimization:
        - `iter_durations`: Tracks the total duration of each iteration.
        - `hyp_opti_durations`: Tracks the duration of hyperparameter optimization.
        - `mean_calc_durations`: Tracks the time spent calculating the predicted mean.
        - `std_calc_durations`: Tracks the time spent calculating the predicted standard deviation.

        Shape:
            - (nb_emg, NB_REP, 1, NB_IT) for all tensors.
            - nb_emg: Number of EMG channels.
            - NB_REP: Number of repetitions per test.
            - NB_IT: Number of iterations.
        """
        # Initialize tensor to store iteration durations
        self.iter_durations = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=torch.float64
        )
        
        # Initialize tensor to store hyperparameter optimization durations
        self.hyp_opti_durations = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=torch.float64
        )
        
        # Initialize tensor to store mean calculation durations
        self.mean_calc_durations = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=torch.float64
        )
        
        # Initialize tensor to store standard deviation calculation durations
        self.std_calc_durations = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT),
            dtype=torch.float64
        )
   
    def initialize_storage_mean_and_std_tensors(self) -> None:
        """Initializes the tensors used to store the predicted mean and standard deviation for each iteration.

        This method sets up two tensors:
        - `P_mean_pred`: To store the predicted mean distribution for each iteration.
        - `P_std_pred`: To store the predicted standard deviation distribution for each iteration.

        These tensors are initialized with zeros and will be progressively filled during the optimization process.

        Shape:
            - (nb_emg, NB_REP, 1, NB_IT, space_size) for both tensors.
            - nb_emg: Number of EMG channels.
            - NB_REP: Number of repetitions per test.
            - NB_IT: Number of iterations.
            - space_size: Size of the search space.
        """
        # Initialize tensor to store predicted mean for each iteration
        self.P_mean_pred = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT, self.space_size),
            dtype=torch.float64
        )
        
        # Initialize tensor to store predicted standard deviation for each iteration
        self.P_std_pred = torch.zeros(
            (self.nb_emg, self.NB_REP, 1, self.NB_IT, self.space_size),
            dtype=torch.float64
        )

    def select_next_query(self, AF: str, gp, train_X: torch.Tensor, train_Y: torch.Tensor, X_test_normed: torch.Tensor) -> int:
        """Selects the next query point based on the acquisition function (AF) applied to the normalized test points.

        Args:
            AF (str): The acquisition function to use. Can be 'EI', 'NEI', 'logEI', 'logNEI', or 'UCB'.
            gp (SingleTaskGP): The Gaussian process model used to predict mean and variance.
            train_X (torch.Tensor): The tensor containing the training inputs (previous queries).
            train_Y (torch.Tensor): The tensor containing the training outputs (responses to previous queries).
            X_test_normed (torch.Tensor): The tensor containing the normalized test points (potential next queries).
                                        Shape: (space_size, space_dim).

        Returns:
            int: The index of the next test point selected based on the acquisition function.
        """
        # Initialize the acquisition function based on the type of AF chosen.
        if AF == 'EI':  # Expected Improvement      
            mu_sample_opt = torch.max(train_Y)  # Find the maximum response (best observed value)
            AF_ = ExpectedImprovement(gp, best_f=mu_sample_opt, maximize=True)

        elif AF == 'NEI':  # Noisy Expected Improvement
            AF_ = NoisyExpectedImprovement(gp, train_X, self.num_fantasies, maximize=True)

        elif AF == 'logEI':  # Log Expected Improvement
            mu_sample_opt = torch.max(train_Y)  # Best observed value (log-transformed)
            AF_ = LogExpectedImprovement(gp, best_f=mu_sample_opt, maximize=True)

        elif AF == 'logNEI':  # Log Noisy Expected Improvement
            AF_ = LogNoisyExpectedImprovement(gp, train_X, self.num_fantasies, maximize=True)

        elif AF == 'UCB':  # Upper Confidence Bound
            AF_ = UpperConfidenceBound(gp, beta=self.KAPPA, maximize=True)

        else:
            raise ValueError(f"Unknown acquisition function: {AF}")

        # Apply the acquisition function to the normalized test points.
        # Reshape X_test_normed for compatibility: (space_size, space_dim) -> (space_size, 1, space_dim)
        ei_val = AF_(X_test_normed[:, None, :])

        # Convert the acquisition values to NumPy format for further processing.
        af_val = ei_val.detach().numpy()

        # Find the index of the maximum acquisition value.
        next_x_idx = np.where(af_val == af_val.max())[0]

        # If multiple points have the same max acquisition value, choose one randomly.
        if next_x_idx.size > 1:
            next_x_idx = np.random.choice(next_x_idx)
        else:
            next_x_idx = next_x_idx[0]

        # Return the index of the next query point.
        return next_x_idx
    
    def get_response(self, emg_i: int, next_x_idx: int, response_type: str = 'valid') -> float:
        """
        Retrieves the response for a given query based on the specified response type.

        Args:
            emg_i (int): The index of the EMG signal.
            next_x_idx (int): The index of the next query point.
            response_type (str, optional): The type of response to retrieve. Choices are 'valid', 'realistic', and 'mean'.
                Defaults to 'valid'.

        Returns:
            float: The response value for the given query.

        Raises:
            ValueError: If the response_type is not one of the valid options ('valid', 'realistic', 'mean').
        """
        if response_type == 'valid':
            resp = self.ds.get_valid_response(emg_i, next_x_idx) # get the response of this query (float)
        elif response_type == 'realistic':
            resp = self.ds.get_realistic_response(emg_i, next_x_idx) # get the response of this query (float)
        elif response_type == 'mean':
            resp = self.ds.get_mean_response(emg_i, next_x_idx) # get the response of this query (float)
        else:
            raise ValueError("response_type is not well defined")
        return(resp)
    
    def npz_save(self, clock_storage: bool, mean_and_std_storage: bool, final: bool = False, emg_i: int = None, r: int = None) -> None:
        """Saves the GPBO results to a .npz file, with optional storage of mean/std predictions and clock durations.

        Args:
            clock_storage (bool): Whether to save iteration and calculation durations.
            mean_and_std_storage (bool): Whether to save mean and standard deviation predictions.
            final (bool, optional): If True, indicates final save. Defaults to False.
            emg_i (int, optional): The index of the current EMG if the simulation is not finished. Required when final is False.
            r (int, optional): The index of the current repetition if the simulation is not finished. Required when final is False.
        """
        save_dict = {
            'P_test_x': self.P_test_x,
            'P_test_x_idx': self.P_test_x_idx,
            'P_test_y': self.P_test_y,
            'best_pred_x': self.best_pred_x,
            'best_pred_x_measured': self.best_pred_x_measured,
            'rand_idx': self.rand_idx,
            'elapsed_time': self.elapsed_time,
        }

        # Add mean and std predictions if mean_and_std_storage is True
        if mean_and_std_storage:
            save_dict.update({
                'P_mean_pred': self.P_mean_pred.detach().numpy(),
                'P_std_pred': self.P_std_pred.detach().numpy(),
            })

        # Add durations if clock_storage is True
        if clock_storage:
            save_dict.update({
                'iter_durations': self.iter_durations,
                'hyp_opti_durations': self.hyp_opti_durations,
                'mean_calc_durations': self.mean_calc_durations,
                'std_calc_durations': self.std_calc_durations,
            })

        # Add state of the simulation
        if final:
            save_dict['state_of_the_simu'] = 'well finished'
        else:
            save_dict['state_of_the_simu'] = f'NOT FINISHED, last save for emg_i={emg_i} and rep={r}'

        # Save the results to the .npz file
        np.savez('results/gpbo_{0}_{1}'.format(self.AF, self.ds.dataset_name, self.name), **save_dict, **self.ds.set)

        if final:
            print("final save of {0} in:   {1}".format(self.name,
                    'results/gpbo_{0}_{1}_{2}'.format(self.AF, self.ds.dataset_name, self.name)))
            
    def erase_storage(self) -> None:
        """Erases stored attributes to free up memory after saving results.

        This method deletes attributes that have been saved to disk in order to 
        release memory. It is useful after a save operation to manage memory usage.

        Returns:
            None: This method does not return a value.
        """
        # deletes attributes to release memory
        del self.P_test_x
        del self.P_test_x_idx
        del self.P_test_y
        del self.best_pred_x
        del self.best_pred_x_measured
        del self.rand_idx
        del self.elapsed_time

        # deletes if exist
        if hasattr(self, 'P_mean_pred'):
            del self.P_mean_pred
        if hasattr(self, 'P_std_pred'):
            del self.P_std_pred

        # deletes if exist
        if hasattr(self, 'iter_durations'):
            del self.iter_durations
        if hasattr(self, 'hyp_opti_durations'):
            del self.hyp_opti_durations
        if hasattr(self, 'mean_calc_durations'):
            del self.mean_calc_durations
        if hasattr(self, 'std_calc_durations'):
            del self.std_calc_durations

    def gpytorch_gpbo(self, emg_i:int, r: int, i: int, response_type: str = 'valid') -> tuple[list, torch.Tensor, torch.Tensor, float, float, float]:
        """Performs a single iteration of Gaussian Process Bayesian Optimization (GPBO).

        This method selects the next query point using either a random or acquisition-based strategy,
        retrieves the corresponding response, updates the training data, and fits a Gaussian Process
        model to the updated data. It also computes the posterior mean and standard deviation for 
        the specified test points.

        Args:
            emg_i (int): The index of the EMG signal being processed.
            r (int): The current round of optimization.
            i (int): The current iteration within the round.
            response_type (str, optional): The type of response to retrieve. Defaults to 'valid'.

        Returns:
            tuple[list, torch.Tensor, torch.Tensor, float, float, float]:
                - query_idx (list): The index of the selected query point.
                - gp_mean_pred (torch.Tensor): The predicted mean values from the Gaussian Process,
                with shape (space_size,).
                - gp_std_pred (torch.Tensor): The predicted standard deviation values from the Gaussian Process,
                with shape (space_size,).
                - hyp_dur (float): Duration of hyperparameter optimization in seconds.
                - mean_dur (float): Duration of mean prediction computation in seconds.
                - std_dur (float): Duration of standard deviation prediction computation in seconds.
        """
        ## ----- Select the next query and get a response ----- ##

        if i < self.NB_RND: # we chose the next node randomly with next_x_idx
            query_idx = self.rand_idx[emg_i, r, i]    
        else: # we have to use an acquisition function to select the next query 
            query_idx = self.select_next_query(AF = self.AF, gp = self.last_used_gp, train_X = self.last_used_train_X,
                                                train_Y = self.last_used_train_Y, X_test_normed = self.X_test_normed)                  
        
        next_x = self.X_test_normed[query_idx]              # entry coordonates for the next query

        resp = self.get_response(emg_i = emg_i, next_x_idx = query_idx, response_type = response_type)



        ## ----- Update tensors ----- ##

        self.P_test_x[emg_i, r, :, i] = next_x             # update the tensor 
        self.P_test_x_idx[emg_i, r, 0, i] = query_idx     # update the tensor
        self.P_test_y[emg_i, r, 0, i] = resp.astype(float) # update the tensor

        train_X = self.P_test_x[emg_i, r, :, :int(i+1)]    # only focus on what have been updated in
                                                    # P_test_x[emg_i, r] (2d-tensor)
        train_X = train_X.T                           # transpose the tensor => shape(2,nb_queries)

        train_Y = self.P_test_y[emg_i, r, 0, :int(i+1)]    # only focus on what have been updated in
                                                    # P_test_y[emg_i, r] (1d-tensor)
        train_Y = train_Y[...,np.newaxis]             # add a new final dimension in the tensor 
                                                    # (2d-tensor) train_y is a column matrix
        if i == 0:
            if resp != 0:
                train_Y = train_Y/train_Y.max() # =1 
        else:
            train_Y = standardize(train_Y) # (data-mean)/std



        ## ----- get and fit the gp ----- ##

        if self.AF == 'NEI' or self.AF == 'logNEI':   
            train_Yvar = torch.full_like(train_Y, 0.15) # create a tensor which has the same shape
                                                        # and type as train_y, but is filled with 
                                                        # a constant value of 0.15.
                                                        # train_Yvar allows to indicate the observed 
                                                        # measurement noise for each measure
            gp = SingleTaskGP(train_X, train_Y, train_Yvar = train_Yvar) # create the GP model
            
        else:
            gp = SingleTaskGP(train_X, train_Y) # create the GP model
        
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # create the log likelihood function 

        tic_hyp = time.perf_counter()
        fit_gpytorch_mll(mll) # optimize the hyperparameters of the GP
        tac_hyp = time.perf_counter()
        hyp_dur = tac_hyp - tic_hyp



        ## ----- get posterior means and std ----- ##

        tic_mean = time.perf_counter() 
        gp_mean_pred = gp.posterior(self.X_test_normed).mean.flatten() # GP prediction for the mean
        tac_mean = time.perf_counter()
        mean_dur = tac_mean - tic_mean

        tic_std = time.perf_counter()
        gp_std_pred = torch.sqrt(gp.posterior(self.X_test_normed).variance.flatten()) # GP prediction for the variance
        tac_std = time.perf_counter()
        std_dur = tac_std - tic_std



        # in order to compute the acquisition function for the next iteration
        self.last_used_gp = gp
        self.last_used_train_X = train_X
        self.last_used_train_Y = train_Y

        return(query_idx, gp_mean_pred, gp_std_pred, hyp_dur, mean_dur, std_dur)
    
    def custum_gpbo(self): #TODO
        pass

    def run_simulations(self, manual_seed: bool = True, clock_storage: bool = True,
                        mean_and_std_storage: bool = False, intermediate_save: bool = False, 
                        response_type: str = 'valid') -> None:
        """
        Execute multiple simulations for Gaussian Process Bayesian Optimization (GPBO) over a defined number of
        electro-myographic (EMG) signals and repetitions.

        Parameters:
        -----------   
        manual_seed : bool
            If True, sets the random seed for reproducibility of results. Default is True.
            
        clock_storage : bool
            If True, initializes storage for timing metrics during the simulation. Default is True.
            
        mean_and_std_storage : bool
            If True, initializes storage for mean and standard deviation tensors from the GP model. Default is False.

        intermediate_save : bool
            If True, save the data collected after each eng loop. Default is False.

        response_type : str
            Can be 'valid', 'realistic' or 'mean'. Find out how to select answers for a particular query. Default is 'valid'. 
                
        Returns:
        --------
        None
            This method does not return a value, but it saves the results in the specified file format.
            
        Notes:
        ------
        - The method uses a progress bar to display the overall progress of the simulations.
        - Various tensors are initialized to store queried locations, predictions, and timing metrics.
        - It allows for both random and acquisition-based querying strategies, determined by the number of iterations.
        - The Gaussian Process model is fit and used to obtain posterior mean and standard deviation predictions.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module='linear_operator.utils.cholesky')
            warnings.filterwarnings("ignore", module='botorch.acquisition.analytic')

            if manual_seed:
                np.random.seed(1) # allow to repeat the same 'random' simu
                torch.manual_seed(1) # allow to repeat the same 'random' simu 

            # we normalize the entry space
            self.X_test_normed = self.normalized_entry_space()  

            # Initialize random indices and storage tensors
            self.rand_idx = self.get_rand_idx()
            self.initialize_storage_basic_tensors()
            if clock_storage:
                self.initialize_storage_clock_tensors()
            if mean_and_std_storage:
                self.initialize_storage_mean_and_std_tensors()

            tic = time.time()

            # Initialize the global progress bar
            with tqdm(total=self.nb_emg*self.NB_REP, desc="Global Progress", unit="iter") as pbar:

                # loop over the different emgs 
                for emg_i in range(self.nb_emg):
                    
                    # loop over the different repetitions 
                    for r in range(self.NB_REP):     
                        
                        queried_loc = [] # initialization of the list of the next_x_idx

                        # loop over the number of node in our grid
                        for i in range(self.NB_IT):

                            tic_it = time.perf_counter()



                            ## ----- GPBO ----- ##

                            last_query_idx, gp_mean_pred, gp_std_pred, hyp_dur, mean_dur, std_dur = self.gpytorch_gpbo(
                                emg_i = emg_i, r = r, i = i, response_type = response_type
                            )

                            if mean_and_std_storage:
                                self.P_mean_pred[emg_i, r, 0, i, :] = gp_mean_pred   # update the tensor 
                                self.P_std_pred[emg_i, r, 0, i, :] = gp_std_pred     # update the tensor



                            ## ----- identify best mean predictions ----- ##

                            self.best_pred_x[emg_i, r, 0, i] = torch.argmax(gp_mean_pred).item() # update the tensor
                            
                            queried_loc.append(last_query_idx)                  # update the list with the last_query_idx we have just chosen
                            not_queried_loc = np.setdiff1d(np.arange(self.space_size), queried_loc)
                            gp_mean_pred[not_queried_loc] = -np.inf # sets to -inf the means of the nodes that have not already been measured
                            self.best_pred_x_measured[emg_i, r, 0, i] = torch.argmax(gp_mean_pred).item() # update the tensor



                            ## ----- update clock metric tensors ----- ##
                        
                            tac_it = time.perf_counter()
                            iter_dur = tac_it - tic_it

                            if clock_storage: 
                                self.iter_durations[emg_i, r, 0, i] = iter_dur
                                self.hyp_opti_durations[emg_i, r, 0, i] = hyp_dur
                                self.mean_calc_durations[emg_i, r, 0, i] = mean_dur
                                self.std_calc_durations[emg_i, r, 0, i] = std_dur
                        


                        # Update the progress bar with the current progress
                        pbar.n = emg_i*self.NB_REP + r + 1
                        pbar.refresh()  # Refresh the progress bar



                    ## ----- intermediate save ----- ##
                    if intermediate_save:
                        if emg_i != int(self.nb_emg - 1):
                            self.npz_save(clock_storage = clock_storage, mean_and_std_storage = mean_and_std_storage, 
                                          emg_i = emg_i, r = r)



        tac = time.time()
        # Calculate and display the elapsed time
        self.elapsed_time = tac - tic
        print(f"Elapsed time: {self.elapsed_time} seconds")



        ## ----- Final save in the npz file ----- ##
        self.npz_save(clock_storage = clock_storage, mean_and_std_storage = mean_and_std_storage, final = True)




