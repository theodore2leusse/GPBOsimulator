import numpy as np
import torch
import gpytorch
from GPcustom.models import GPytorchModel
from botorch.utils.transforms import standardize
import time

class QueriesInfo:
    def __init__(self, space_shape):
        self.space_shape = space_shape
        self.query_map = np.full(space_shape, 0)
        self.mean_map = np.full(space_shape, np.nan)
        self.var_map = np.full(space_shape, np.nan)
        
    def update_map(self, query_x, query_y):
        self.query_map[query_x] += 1
        if self.query_map[query_x] == 1:
            self.mean_map[query_x] = query_y
        elif self.query_map[query_x] == 2:
            new_mean_map = (query_y + self.mean_map[query_x]) / 2
            self.var_map[query_x] = ((query_y-new_mean_map)**2 + (self.mean_map[query_x]-new_mean_map)**2) / 2
            self.mean_map[query_x] = new_mean_map
        else:
            new_mean_map = ((self.query_map[query_x]-1)*self.mean_map[query_x] + query_y) / self.query_map[query_x]
            self.var_map[query_x] = ((self.query_map[query_x]-1) * (self.var_map[query_x] + (new_mean_map-self.mean_map[query_x])**2) + (query_y-self.mean_map[query_x])**2) / self.query_map[query_x]
            self.mean_map[query_x] = new_mean_map

    def mat2vec(self, mat, ch2xy):
        vec = np.zeros(len(ch2xy))
        for i in range(len(ch2xy)):
            vec[i] = mat[*(ch2xy[i]-1)]
        return vec

    def idx2coord(self, idx):
        coord = []
        for i in range(len(idx)):
            coord.append(idx[i]/(self.space_shape[i]-1))
        return tuple(coord)
    
    def coord2idx(self, coord):
        idx = []
        for i in range(len(coord)):
            idx.append(round(coord[i]*(self.space_shape[i]-1)))
        return tuple(idx)
    
    def get_mean_queries(self):
        mean_queries_list = []
        for idx in np.ndindex(self.space_shape):
            if not np.isnan(self.mean_map[idx]):
                mean_queries_list.append([self.idx2coord(idx), self.mean_map[idx]])
        return mean_queries_list
    
    def estimate_HP(self, outputscale: float = None, noise: float = None, max_iters_training_gp: int = 100):
        mean_queries = self.get_mean_queries()
        train_x = torch.tensor([list(item[0]) for item in mean_queries], dtype=torch.float64)
        train_y = torch.tensor([item[1] for item in mean_queries], dtype=torch.float64)

        if np.sum(self.query_map) == 1:
            train_y = torch.tensor([1.], dtype=torch.float64)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp = GPytorchModel(
                        train_x=train_x,
                        train_y=train_y,
                        likelihood=self.likelihood,
                        kernel_type='Matern52',
                        outputscale=None, 
                        noise=None
                    )
        else:
            train_y = standardize(train_y)
            self.gp.set_train_data(
                train_x,
                train_y,
                strict=False,
            )

        self.gp.double()

        # Find optimal model hyperparameters
        self.gp.train_model(train_x, train_y, max_iters=max_iters_training_gp, lr=0.1, Verbose=False)
        self.hyperparams = self.gp.get_hyperparameters()

    def predict(self, test_X, ch2xy):
        """l'idée serait de faire une méthode predict où pred_std = self.gp.std/(self.query_map+1)"""
        gp_mean_pred, gp_std_pred = self.gp.predict(test_X)
        gp_std_pred = gp_std_pred/(self.mat2vec(self.query_map, ch2xy) + 1)

        return gp_mean_pred, gp_std_pred
        
