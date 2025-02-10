import torch
import numpy as np
import math
import gpytorch

from GPcustom.models import FixedGP, GPytorchFixed

def create_space_test(shape):
    coords = [torch.linspace(0, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
    return grid.reshape(-1, len(shape))

class NNasOnlineGP():

    def __init__(self, model, space_shape, ch2xy):
        self.model = model
        self.model.eval()
        self.space_shape = space_shape
        self.ch2xy = ch2xy
        self.nb_queries = 0
        self.mean_queries = 0
        self.variance_queries = 0 

        self.space_test = create_space_test(space_shape)

    def mat2vec(self, mat):
        vec = np.zeros(len(self.ch2xy))
        for i in range(len(self.ch2xy)):
            vec[i] = mat[*(self.ch2xy[i]-1)]
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
    
    def get_input_query(self, query_x, query_y, lengthscale=[0.3, 0.3]):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        elem_gp = GPytorchFixed(
                        train_x=self.idx2coord(query_x), train_y=torch.tensor([1.]), 
                        likelihood=likelihood, kernel_type='Matern52', noise=0.05, 
                        outputscale=1, lengthscale=lengthscale)
        elem_mean, elem_std = elem_gp.predict(self.X_test_normed_tensor)
        elem_gp.double()
        #TODO predire sur self.space_test et obtenir next_input_query_x et next_input_query_y
    
    def next_query(self, query_x, query_y):
        ## --- Calculate new mean & var --- ##
        if self.nb_queries == 0:
            new_mean_queries = query_y
            new_variance_queries = 0
        else:
            new_mean_queries = ((self.nb_queries*self.mean_queries + query_y)
                                /(self.nb_queries+1))
            new_variance_queries = (
                (self.nb_queries * (self.variance_queries 
                                    + (new_mean_queries-self.mean_queries)**2) 
                 + (query_y-new_mean_queries)**2) 
                / (self.nb_queries+1)
            )
        ## --- store info for next prediction --- ##
        if self.nb_queries == 1:
            if query_y > self.last_query_y:
                self.next_input_mean = -self.mean
            else:
                self.next_input_mean = self.mean
            self.next_input_std = self.std
        elif self.nb_queries > 1:
            self.next_input_mean = ((math.sqrt(self.variance_queries) * self.mean 
                                     + self.mean_queries - new_mean_queries) 
                                    / math.sqrt(new_variance_queries))
            self.next_input_std = self.std
        self.next_input_query_x = None #TODO 
        self.next_input_query_y = (query_y - new_mean_queries) / math.sqrt(new_variance_queries) * None #TODO
        self.last_query_x = query_x
        self.last_query_y = query_y
        ## --- Update of nb, mean & var --- ##
        self.mean_queries = new_mean_queries
        self.variance_queries = new_variance_queries
        self.nb_queries += 1

    def predict(self, test_X):
        if self.nb_queries == 1:
            output = None
        else:
            input = None # doit etre de la shape (1,4,L,l)
            output = self.model(*torch.split(input, 1, dim=1))
        self.mean = output[0]
        self.std = output[1]
        return self.mean, self.std
