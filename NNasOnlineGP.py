import torch 
import numpy as np 
import math
import gpytorch

from GPcustom.models import GPytorchFixed

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

        self.grid = create_space_test(space_shape)
    
    def from_image_to_vec_for_BO(self, map):
        vec = torch.full((len(self.ch2xy),), np.nan)
        for i in range(len(self.ch2xy)):
            vec[i] = map[0, 0, self.ch2xy[i,0]-1, self.ch2xy[i,1]-1]
        return vec

    def get_elem_mean_and_std_fixed_lengthscales(self, kernel_type='Matern52', 
                                    noise=0.05, outputscale=1, lengthscale=[0.3, 0.3]):
        self.all_maps_elem_mean = torch.full((len(self.grid),1 , 1, self.space_shape[0], self.space_shape[1]), np.nan)
        self.all_maps_elem_std = torch.full((len(self.grid),1 , 1, self.space_shape[0], self.space_shape[1]), np.nan)

        for i in range(len(self.grid)):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            elem_gp = GPytorchFixed(
                            train_x=self.grid[i:i+1], train_y=torch.tensor([1.]), 
                            likelihood=likelihood, kernel_type=kernel_type, noise=noise, 
                            outputscale=outputscale, lengthscale=lengthscale)
            elem_gp.double()
            elem_mean, elem_std = elem_gp.predict(self.grid)
            for j in range(len(self.grid)):
                k, l = int(self.grid[j,0]*9 + 0.01), int(self.grid[j,1]*9 + 0.01)
                self.all_maps_elem_mean[i, 0, 0, k, l] = elem_mean[j].detach().item()
                self.all_maps_elem_std[i, 0, 0, k, l] = elem_std[j].detach().item()
    
    def get_input_query(self, query_x, query_y, kernel_type='Matern52', 
                                    noise=0.05, outputscale=1, lengthscale=[0.3, 0.3]):
        if hasattr(self, 'all_maps_elem_mean'):
            k, l = int(query_x[0]*9 + 0.01), int(query_x[1]*9 + 0.01)
            id_grid = k*10 + l
            maps_elem_mean = self.all_maps_elem_mean[id_grid]
            maps_elem_std = self.all_maps_elem_std[id_grid]
            return maps_elem_mean*query_y, maps_elem_std
        else:
            maps_elem_mean = torch.full((1 , 1, self.space_shape[0], self.space_shape[1]), np.nan)
            maps_elem_std = torch.full((1 , 1, self.space_shape[0], self.space_shape[1]), np.nan)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            elem_gp = GPytorchFixed(
                            train_x=self.idx2coord(query_x), train_y=torch.tensor([1.]), 
                            likelihood=likelihood, kernel_type='Matern52', noise=0.05, 
                            outputscale=1, lengthscale=lengthscale)
            elem_gp.double()
            elem_mean, elem_std = elem_gp.predict(self.grid)
            for j in range(len(self.grid)):
                k, l = int(self.grid[j,0]*9 + 0.01), int(self.grid[j,1]*9 + 0.01)
                maps_elem_mean[0, 0, k, l] = elem_mean[j].detach().item()
                maps_elem_std[0, 0, k, l] = elem_std[j].detach().item()
            return maps_elem_mean*query_y, maps_elem_std
    
    def update_with_query(self, query_x, query_y):
        if self.nb_queries == 0:
            new_mean_queries = query_y
            new_variance_queries = 0
            self.img_mean, self.img_std = self.get_input_query(query_x=query_x, query_y=1.,
                kernel_type='Matern52', noise=0.05, outputscale=1, lengthscale=[0.3, 0.3])
        else:
            new_mean_queries = ((self.nb_queries*self.mean_queries + query_y)
                                /(self.nb_queries+1))
            new_variance_queries = (
                (self.nb_queries * (self.variance_queries + (new_mean_queries-self.mean_queries)**2) 
                 + (query_y-new_mean_queries)**2) 
                / (self.nb_queries+1)
            )
            modified_query_y = (query_y - new_mean_queries) / math.sqrt(new_variance_queries)
            elem_mean_input, elem_std_input = self.get_input_query(query_x=query_x, query_y=modified_query_y,
                kernel_type='Matern52', noise=0.05, outputscale=1, lengthscale=[0.3, 0.3])
            input = torch.cat((self.img_mean, self.img_std, elem_std_input, elem_mean_input), dim=1)
            output = self.model(*torch.split(input, 1, dim=1))
            self.img_mean = output[:,0:1]
            self.img_std = output[:,1:2]

        self.mean_queries = new_mean_queries
        self.variance_queries = new_variance_queries
        self.nb_queries += 1

        self.mean = self.from_image_to_vec_for_BO(self.img_mean).detach()
        self.std = self.from_image_to_vec_for_BO(self.img_std).detach()

        return(self.mean.clone(), self.std.clone())