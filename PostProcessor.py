# sciNeurotech Lab 
# Theodore
"""
in this file, we will define a class in order to process data 
"""

# import
import numpy as np
import matplotlib.pyplot as plt


class PostProcessor():
    """
    class to post process data

    Attributes:
        path_to_npz_file (str): The file path to the npz file containing the dataset.
        dataset_name (str): The name of the dataset being used.
        data (dict(np.array)): npz file contents, containing the saved experiment data.
        P_test_x (np.array): input coordinates of the queries during the simulation, 
                            shaped as (nb_emg, NB_REP, space_dim, nb_it).
        P_test_y (np.array): Responses to the queries, shaped as (nb_emg, NB_REP, 1, nb_it).
        P_mean_pred (np.array): Mean predictions for each iteration, 
                                shaped as (nb_emg, NB_REP, 1, nb_it, space_size).
        P_std_pred (np.array): Standard deviation of the predictions for each iteration, 
                            shaped as (nb_emg, NB_REP, 1, nb_it, space_size).
        best_pred_x (np.array): Electrode ID of the best prediction for each iteration, 
                                shaped as (nb_emg, NB_REP, 1, nb_it).
        best_pred_x_measured (np.array): Electrode ID of the best prediction including 
                                        only electrodes that have already been measured, 
                                        shaped as (nb_emg, NB_REP, 1, nb_it).
        rand_idx (np.array): records random choices of queries for the first iterations of optimization
        ds_set (dict(np.array)): corresponds to the DataSet.set of the dataset used in the simulation
        nb_emg (int): nb of emgs used in the simulation
        NB_REP (int): nb of repetitions in the simulation
        nb_it (int): nb of iteration for each optimization

    Methods:
        load_data():
            load the data of the npz file
        exploration(emgs_idx: list[int] = None, REP_idx: list[int] = None, status: str = 'offline'):
            Calculate the exploration metric
        exploitation(emgs_idx: list[int] = None, REP_idx: list[int] = None, status: str = 'offline'):
            Calculate the exploitation metric
    """

    def __init__(self, path_to_npz_file: str, dataset_name: str = 'NO_NAME') -> None:
        """
        initialize a DataProcess instance

        Args:
            path_to_npz_file (str): Path to the npz file we want to analyse
            dataset_name (str, optional): Name for the data. Defaults to 'NO_NAME'.
        """
        self.path_to_npz_file = path_to_npz_file
        self.dataset_name = dataset_name
        self.data = {}

    def load_data(self, load_mean: bool = False, load_std: bool = False, load_durations = False) -> None:
        """
        load the data of the npz file 

        Args:
            load_mean (bool, optional): True if you want to load the mean for each it. Defaults to 'False'.
            load_std (bool, optional): True if you want to load the std for each it. Defaults to 'False'.
            load_durations (bool, optional): True if you want to load the different durations for each it. Defaults to 'False'.

        Update:
            SET (dict): dict_keys(['emgs', 'nChan', 'sorted_isvalid', 'sorted_resp',
                                   'sorted_respMean', 'ch2xy'])
                - emgs: 1 x e cell array of strings. Muscle names for each implanted EMG.
                - nChan: a single scalar equal to c. Number of cortical array channels.
                - sorted_resp: c x e cell array. Corresponds to "response"*, where each EMG 
                  response has been sorted and assigned to the source stimulation site.
                - sorted_isvalid: c x e cell array. Corresponds to "isvalid"*, where each EMG
                  response has been sorted and assigned to the source stimulation site.
                - sorted_respMean: c x e single array. Average of all valid responses, 
                  segregated per stimulating channel and per EMG.
                - ch2xy: c x 2 matrix with <x,y> relative coordinates for each stimulation 
                  channel. Units are intra-electrode spacing.

                * response: 1 x e cell array. Each entry is a numerical matrix associated to a 
                  single EMG. Each entry is j x 1 and represents a sampled cumulative response 
                  (during peak activity) for each evoked_emg. Thus, each trace is collapsed to 
                  a single outcome value.
                * isvalid : 1 x e cell array. Each entry is a numerical matrix associated to a 
                  single EMG. Each entry is j x 1 and determines whether the recorded response 
                  can be considered valid. A value of 1 means that we found no reason to exclude 
                  the response. A value of 0 means that baseline (pre-stimulus) activity exceeds 
                  accepted levels and indicates that spontaneous EMG activity was ongoing at the 
                  time of stimulus delivery. A value of -1 indicates that the response is an 
                  outlier, yet baseline activity is within range. We consider the "0" and "-1" 
                  categories practically possible and impossible to reject during online trials, 
                  respectively.

                c : number of channels in the implanted cortical array (variable between species).
                e : implanted EMG count (variable between subjects). 
                j : individual stimuli count throughout the session. 
                t : number of recorded samples for each stimulus. 
                Sampling frequency is sampFreqEMG (variable between species)
        """
        self.data = np.load(self.path_to_npz_file, allow_pickle = True) # TODO ???
        # Access the different variables that were saved
        self.P_test_x = self.data['P_test_x']            # Tensor storing queries' input coordinates 
        self.P_test_x_idx = self.data['P_test_x_idx']    # Tensor storing queries' input idx
        self.P_test_y = self.data['P_test_y']            # Tensor storing the corresponding responses to the queries
        if load_mean: 
            self.P_mean_pred = self.data['P_mean_pred']  # Tensor storing mean in the search space after each iteration 
        if load_std:
            self.P_std_pred = self.data['P_std_pred']    # Tensor storing standard deviation in the search space after each iteration 
        self.best_pred_x = self.data['best_pred_x']      # Tensor listing the  node_id where is the best predictions ; shape = (nb_emg, nb_rep, 1, nb_it)
        self.best_pred_x_measured = self.data['best_pred_x_measured'] # Tensor listing the  node_id where is the best predictions, including only measured nodes ; shape = (nb_emg, nb_rep, 1, nb_it)
        self.rand_idx = self.data['rand_idx']            # Random indices, possibly used for shuffling or sampling
        self.ds_set = {key: self.data[key] for key in 
                ['emgs', 'nChan', 'sorted_isvalid', 'sorted_resp', 'sorted_respMean', 'ch2xy']}
        self.nb_emg = self.P_test_x_idx.shape[0]         # nb of emgs used in the simulation
        self.NB_REP = self.P_test_x_idx.shape[1]         # nb of repetitions used in the simulation
        self.nb_it = self.P_test_x_idx.shape[3]          # nb of iterations used in the simulation
        if load_durations:
            self.elapsed_time = self.data['elapsed_time']               # save the simulation duration
            self.iter_durations = self.data['iter_durations']           # save the iteration duration for each it
            self.hyp_opti_durations = self.data['hyp_opti_durations']   # save the hyperparameters' optimization duration for each it
            self.mean_calc_durations = self.data['mean_calc_durations'] # save the mean calculation duration for each it
            self.std_calc_durations = self.data['std_calc_durations']   # save the std calculation duration for each it

    def exploration(self, emgs_idx: list[int] = None, REP_idx: list[int] = None, status: str = 'offline') ->  np.ndarray:
        """
        Calculate the exploration metric

        Args:
            emgs_idx (list[int], optional): list of the emgs you want to consider to calculate the metric. Defaults to None what correspond to all the emgs.
            REP_idx (list[int], optional): list of the repitutions you want to consider to calculate the metric. Defaults to None what correspond to all the repetitions.

        Returns:
            np.ndarray: mean of the exploration metric for each iteration.
        """
        if emgs_idx is None:
            emgs_idx = list(range(self.nb_emg))
        if REP_idx is None:
            REP_idx = list(range(self.NB_REP))

        perf_explore = np.zeros((len(emgs_idx), len(REP_idx), self.nb_it))

        emg_i_id = 0
        for emg_i in emgs_idx:
            rep_id = 0
            for rep in REP_idx:
                for it in range(self.nb_it): 
                    best_x = self.best_pred_x_measured[emg_i, rep, 0, it] # best prediction's electrode_id that has 
                                                                          # already been measured (int)
                    if status == 'online':
                        perf_explore[emg_i_id, rep_id, it] = (np.max(self.P_mean_pred[emg_i, rep, 0, it, :]) *
                                                              self.ds_set['sorted_respMean'][best_x, emg_i]) 
                    elif status == 'offline':
                        perf_explore[emg_i_id, rep_id, it] = ((self.ds_set['sorted_respMean'][best_x, emg_i] - 
                                                               np.min(self.ds_set['sorted_respMean'][:, emg_i])) /
                                                              (np.max(self.ds_set['sorted_respMean'][:, emg_i]) -
                                                               np.min(self.ds_set['sorted_respMean'][:, emg_i]))) # fill the array 
                rep_id += 1
            emg_i_id += 1
        return(np.mean(perf_explore, axis = (0,1)))
                    
    def exploitation(self, emgs_idx: list[int] = None, REP_idx: list[int] = None, status: str = 'offline') ->  np.ndarray:
        """
        Calculate the exploitation metric

        Args:
            emgs_idx (list[int], optional): list of the emgs you want to consider to calculate the metric. Defaults to None what correspond to all the emgs.
            REP_idx (list[int], optional): list of the repitutions you want to consider to calculate the metric. Defaults to None what correspond to all the repetitions.

        Returns:
            np.ndarray: mean of the exploitation metric for each iteration.
        """
        if emgs_idx is None:
            emgs_idx = list(range(self.nb_emg))
        if REP_idx is None:
            REP_idx = list(range(self.NB_REP))

        perf_exploit = np.zeros((len(emgs_idx), len(REP_idx), self.nb_it))

        emg_i_id = 0 
        for emg_i in emgs_idx:
            rep_id = 0 
            for rep in REP_idx:
                for it in range(self.nb_it): 
                    curr_x = self.P_test_x_idx[emg_i, rep, 0, it] # last electrode_id evaluated (int)
                    best_x = self.best_pred_x_measured[emg_i, rep, 0, it] # best prediction's electrode_id that has 
                                                                          # already been measured (int)
                    if status == 'online':
                        perf_exploit[emg_i_id, rep_id, it] = (self.P_mean_pred[emg_i, rep, 0, it, curr_x] * 
                                                              self.ds_set['sorted_respMean'][best_x, emg_i])
                    elif status =='offline':          
                        perf_exploit[emg_i_id, rep_id, it] = ((self.ds_set['sorted_respMean'][curr_x, emg_i] - 
                                                               np.min(self.ds_set['sorted_respMean'][:, emg_i])) /
                                                              (np.max(self.ds_set['sorted_respMean'][:, emg_i]) -
                                                               np.min(self.ds_set['sorted_respMean'][:, emg_i]))) # fill the array
                rep_id += 1
            emg_i_id += 1
        return(np.mean(perf_exploit, axis = (0,1)))
    
    def duration_metrics(self, emgs_idx: list[int] = None, REP_idx: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the duration metrics. This metric consists of averaging the various calculation 
        times over emgs and repetitions. 

        Args:
            emgs_idx (list[int], optional): list of the emgs you want to consider to calculate the 
                                            metric. Defaults to None what correspond to all the emgs.
            REP_idx (int, optional): the repetition you want to consider to calculate 
                                    the metric. Defaults to None what correspond to all the 
                                    repetitions.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                A tuple containing four numpy arrays, each with shape (nb_it):
                    - perf_iter (np.ndarray): The mean duration of each iteration.
                    - perf_hyp (np.ndarray): The mean duration of the hyperparameter optimization.
                    - perf_mean (np.ndarray): The mean duration of mean calculation.
                    - perf_std (np.ndarray): The mean duration of standard deviation calculation.
        """
        if emgs_idx is None:
            emgs_idx = list(range(self.nb_emg))
        if REP_idx is None:
            perf_iter = np.mean(self.iter_durations[emgs_idx, :, :, :], axis = (0,1,2))      
            perf_hyp = np.mean(self.hyp_opti_durations[emgs_idx, :, :, :], axis = (0,1,2))
            perf_mean = np.mean(self.mean_calc_durations[emgs_idx, :, :, :], axis = (0,1,2))
            perf_std = np.mean(self.std_calc_durations[emgs_idx, :, :, :], axis = (0,1,2))
        else:
            perf_iter = np.mean(self.iter_durations[emgs_idx, REP_idx, :, :], axis = (0,1,2))      
            perf_hyp = np.mean(self.hyp_opti_durations[emgs_idx, REP_idx, :, :], axis = (0,1,2))
            perf_mean = np.mean(self.mean_calc_durations[emgs_idx, REP_idx, :, :], axis = (0,1,2))
            perf_std = np.mean(self.std_calc_durations[emgs_idx, REP_idx, :, :], axis = (0,1,2))
        return(perf_iter, perf_hyp, perf_mean, perf_std)




