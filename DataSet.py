# sciNeurotech Lab 
# Theodore
"""
in this file, we will define a class in order to process data 
"""

# import
import numpy as np
from scipy.io import loadmat

class DataSet():
    """
    class to process data

    Attributes:
        path_to_dataset_folder (str): path to the dataset folder where there is the
                                        file you want to process
        dataset_type (str): 'nhp' (non human primat) or 'rat' 
        dataset_file (str): name of the file you want the data to be processed
        dataset_name (str): name you want to give to the dataset
        set (dict): dict_keys(['emgs', 'nChan', 'sorted_isvalid', 'sorted_resp',
                               'sorted_respMean', 'ch2xy']) 

    Methods:
        load_matlab_data():
            function from Rose
            use a .mat files and return a dictionary
        get_valid_response(emg_id: int, electrode_id: int):
            get one of the valid responses of one electrode measured by one emg
        get_realistic_response(emg_id: int, electrode_id: int):
            get one of the valid and outlier responses of one electrode measured by one emg
        get_mean_response(emg_id: int, electrode_id: int):
            the mean response of one electrode measured by one emg
    """

    def __init__(self, path_to_dataset_folder: str, dataset_type: str, dataset_file: str, dataset_name: str = 'NO_NAME') -> None:
        """
        initialize a DataProcess instance

        Args:
            path_to_dataset_folder (str): path to the dataset folder where there is the
                                          file you want to process
            dataset_type (str): 'nhp' (non human primat) or 'rat' 
            dataset_file (str): name of the file you want to process
            dataset_name (str, optional): name of the dataset. Defaults to 'NO_NAME'.
        """
        self.path_to_dataset_folder = path_to_dataset_folder
        self.dataset_type = dataset_type
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.set = {}

    def load_matlab_data(self) -> None: 
        """        
        function from Rose
        use a .mat files and return a dictionary 

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
                Sampling frequency is sampFreqEMG (variable between species).  
        """
        # The order of the variables inside the dict changes between Macaque, Cebus and rats
        if self.dataset_type == 'nhp':
            if self.dataset_file == 'Cebus1_M1_190221.mat':
                Cebus1_M1_190221 = loadmat(self.path_to_dataset_folder+'/Cebus1_M1_190221.mat')
                Cebus1_M1_190221 = {'emgs': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][0][0],
                'nChan': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][2][0][0],
                'sorted_isvalid': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][8],
                'sorted_resp': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][9],
                'sorted_respMean': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][10],
                'ch2xy': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][16]}
                self.set = Cebus1_M1_190221
            if self.dataset_file == 'Cebus2_M1_200123.mat':
                Cebus2_M1_200123 = loadmat(self.path_to_dataset_folder+'/Cebus2_M1_200123.mat')  
                Cebus2_M1_200123 = {'emgs': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][0][0],
                'nChan': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][2][0][0],
                'sorted_isvalid': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][8],
                'sorted_resp': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][9],
                'sorted_respMean': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][10],
                'ch2xy': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][16]}
                self.set = Cebus2_M1_200123
            if self.dataset_file == 'Macaque1_M1_181212.mat':    
                Macaque1_M1_181212 = loadmat(self.path_to_dataset_folder+'/Macaque1_M1_181212.mat')
                Macaque1_M1_181212 = {'emgs': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][0][0],
                'nChan': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][2][0][0],
                'sorted_isvalid': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][8],
                'sorted_resp': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][9],              
                'sorted_respMean': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][15],
                'ch2xy': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][14]}            
                self.set = Macaque1_M1_181212
            if self.dataset_file == 'Macaque2_M1_190527.mat':    
                Macaque2_M1_190527 = loadmat(self.path_to_dataset_folder+'/Macaque2_M1_190527.mat')
                Macaque2_M1_190527 = {'emgs': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][0][0],
                'nChan': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][2][0][0],
                'sorted_isvalid': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][8],
                'sorted_resp': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][9],              
                'sorted_respMean': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][15],
                'ch2xy': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][14]}
                self.set = Macaque2_M1_190527
        elif self.dataset_type == 'rat':   
            if self.dataset_file == 'rat1_M1_190716.mat':
                rat1_M1_190716 = loadmat(self.path_to_dataset_folder+'/rat1_M1_190716.mat')
                rat1_M1_190716 = {'emgs': rat1_M1_190716['rat1_M1_190716'][0][0][0][0],
                'nChan': rat1_M1_190716['rat1_M1_190716'][0][0][2][0][0],
                'sorted_isvalid': rat1_M1_190716['rat1_M1_190716'][0][0][8],
                'sorted_resp': rat1_M1_190716['rat1_M1_190716'][0][0][9],              
                'sorted_respMean': rat1_M1_190716['rat1_M1_190716'][0][0][15],
                'ch2xy': rat1_M1_190716['rat1_M1_190716'][0][0][14]}            
                self.set = rat1_M1_190716
            if self.dataset_file == 'rat2_M1_190617.mat':
                rat2_M1_190617 = loadmat(self.path_to_dataset_folder+'/rat2_M1_190617.mat')
                rat2_M1_190617 = {'emgs': rat2_M1_190617['rat2_M1_190617'][0][0][0][0],
                'nChan': rat2_M1_190617['rat2_M1_190617'][0][0][2][0][0],
                'sorted_isvalid': rat2_M1_190617['rat2_M1_190617'][0][0][8],
                'sorted_resp': rat2_M1_190617['rat2_M1_190617'][0][0][9],              
                'sorted_respMean': rat2_M1_190617['rat2_M1_190617'][0][0][15],
                'ch2xy': rat2_M1_190617['rat2_M1_190617'][0][0][14]}         
                self.set = rat2_M1_190617          
            if self.dataset_file == 'rat3_M1_190728.mat':
                rat3_M1_190728 = loadmat(self.path_to_dataset_folder+'/rat3_M1_190728.mat')
                rat3_M1_190728 = {'emgs': rat3_M1_190728['rat3_M1_190728'][0][0][0][0],
                'nChan': rat3_M1_190728['rat3_M1_190728'][0][0][2][0][0],
                'sorted_isvalid': rat3_M1_190728['rat3_M1_190728'][0][0][8],
                'sorted_resp': rat3_M1_190728['rat3_M1_190728'][0][0][9],              
                'sorted_respMean': rat3_M1_190728['rat3_M1_190728'][0][0][15],
                'ch2xy': rat3_M1_190728['rat3_M1_190728'][0][0][14]}           
                self.set = rat3_M1_190728                       
            if self.dataset_file == 'rat4_M1_191109.mat':
                rat4_M1_191109 = loadmat(self.path_to_dataset_folder+'/rat4_M1_191109.mat')
                rat4_M1_191109 = {'emgs': rat4_M1_191109['rat4_M1_191109'][0][0][0][0],
                'nChan': rat4_M1_191109['rat4_M1_191109'][0][0][2][0][0],
                'sorted_isvalid': rat4_M1_191109['rat4_M1_191109'][0][0][8],
                'sorted_resp': rat4_M1_191109['rat4_M1_191109'][0][0][9],              
                'sorted_respMean': rat4_M1_191109['rat4_M1_191109'][0][0][15],
                'ch2xy': rat4_M1_191109['rat4_M1_191109'][0][0][14]}            
                self.set = rat4_M1_191109                       
            if self.dataset_file == 'rat5_M1_191112.mat':
                rat5_M1_191112 = loadmat(self.path_to_dataset_folder+'/rat5_M1_191112.mat')
                rat5_M1_191112 = {'emgs': rat5_M1_191112['rat5_M1_191112'][0][0][0][0],
                'nChan': rat5_M1_191112['rat5_M1_191112'][0][0][2][0][0],
                'sorted_isvalid': rat5_M1_191112['rat5_M1_191112'][0][0][8],
                'sorted_resp': rat5_M1_191112['rat5_M1_191112'][0][0][9],              
                'sorted_respMean': rat5_M1_191112['rat5_M1_191112'][0][0][15],
                'ch2xy': rat5_M1_191112['rat5_M1_191112'][0][0][14]}           
                self.set = rat5_M1_191112                      
            if self.dataset_file == 'rat6_M1_200218.mat':
                rat6_M1_200218 = loadmat(self.path_to_dataset_folder+'/rat6_M1_200218.mat')        
                rat6_M1_200218 = {'emgs': rat6_M1_200218['rat6_M1_200218'][0][0][0][0],
                'nChan': rat6_M1_200218['rat6_M1_200218'][0][0][2][0][0],
                'sorted_isvalid': rat6_M1_200218['rat6_M1_200218'][0][0][8],
                'sorted_resp': rat6_M1_200218['rat6_M1_200218'][0][0][9],              
                'sorted_respMean': rat6_M1_200218['rat6_M1_200218'][0][0][15],
                'ch2xy': rat6_M1_200218['rat6_M1_200218'][0][0][14]}          
                self.set = rat6_M1_200218               
        else:
            print('Invalid value for dataset variable. Has to be either \'nhp\' or \'rat\'. ')     
            self.set = None

    def get_valid_response(self, emg_id: int, electrode_id: int) -> float:
        """
        get one of the valid responses of one electrode measured by one emg

        Args:
            emg_id (int): emg identifier
            electrode_id (int): electrode identifier

        Returns:
            resp (double): response to one of the query made in electrode_id and measured by emg_id
        """
        valid_resp = self.set['sorted_resp'][electrode_id, emg_id][:,0][
            self.set['sorted_isvalid'][electrode_id, emg_id][:,0] == 1]    
        resp = np.random.choice(valid_resp) # select randomly one repetition
        return resp  
    
    def get_realistic_response(self, emg_id: int, electrode_id: int) -> float:
        """
        get one of the valid and outlier responses of one electrode measured by one emg

        Args:
            emg_id (int): emg identifier
            electrode_id (int): electrode identifier

        Returns:
            resp (double): response to one of the query made in electrode_id and measured by emg_id
        """
        realistic_resp = self.set['sorted_resp'][electrode_id, emg_id][:, 0][
            (self.set['sorted_isvalid'][electrode_id, emg_id][:, 0] == 1) | 
            (self.set['sorted_isvalid'][electrode_id, emg_id][:, 0] == -1)]    
        resp = np.random.choice(realistic_resp) # select randomly one repetition
        return resp  

    def get_mean_response(self, emg_id: int, electrode_id: int) -> float:
        """
        get the mean response of one electrode measured by one emg

        Args:
            emg_id (int): emg identifier
            electrode_id (int): electrode identifier

        Returns:
            resp (double): response the mean response for electrode_id and measured by emg_id
        """
        resp = self.set['sorted_respMean'][electrode_id, emg_id] 
        return resp 
    

if __name__ == "__main__":
    ds = DataSet('data/','nhp','Cebus1_M1_190221.mat')
    ds.load_matlab_data()
    print(ds.set.keys())