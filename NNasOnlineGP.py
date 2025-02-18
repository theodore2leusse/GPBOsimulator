# sciNeurotech Lab 
# Theodore

import torch 
import numpy as np 
import math
import gpytorch

from GPcustom.models import GPytorchFixed

"""
This module contains the NNasOnlineGP class for performing online surrogate model (inspiration from Gaussian Process (GP)) regression with a neural network model.
"""

def create_space_test(shape):
    coords = [torch.linspace(0, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
    return grid.reshape(-1, len(shape))

class NNasOnlineGP():
    """
    NNasOnlineGP is a class for performing online Gaussian Process (GP) regression with a neural network model.
    Attributes:
        model (torch.nn.Module): The neural network model used for predictions.
        space_shape (tuple): The shape of the input space.
        ch2xy (np.ndarray): A mapping from channels to x, y coordinates.
        nb_queries (int): The number of queries made.
        mean_queries (float): The mean of the queries.
        variance_queries (float): The variance of the queries.
        grid (torch.Tensor): The grid of test points in the input space.
        all_maps_elem_mean (torch.Tensor): The mean of the GP predictions for all grid points.
        all_maps_elem_std (torch.Tensor): The standard deviation of the GP predictions for all grid points.
        img_mean (torch.Tensor): The mean image from the GP predictions.
        img_std (torch.Tensor): The standard deviation image from the GP predictions.
        mean (torch.Tensor): The mean vector for Bayesian Optimization.
        std (torch.Tensor): The standard deviation vector for Bayesian Optimization.
    Methods:
        __init__(self, model, space_shape, ch2xy):
            Initializes the NNasOnlineGP class with the given model, space shape, and channel to x, y mapping.
        from_image_to_vec_for_BO(self, map):
            Converts an image map to a vector for Bayesian Optimization.
        get_elem_mean_and_std_fixed_lengthscales(self, kernel_type='Matern52', noise=0.05, outputscale=1, lengthscale=[0.3, 0.3]):
            Computes the mean and standard deviation of the GP predictions for all grid points with fixed lengthscales.
        get_input_query(self, query_x, query_y, kernel_type='Matern52', noise=0.05, outputscale=1, lengthscale=[0.3, 0.3]):
            Gets the input query for the GP model and returns the mean and standard deviation of the predictions.
        update_with_query(self, query_x, query_y):
            Updates the GP model with a new query and returns the updated mean and standard deviation vectors for Bayesian Optimization.
    """

    def __init__(self, model, space_shape, ch2xy):
        """
        Initializes the NNasOnlineGP class with the given model, space shape, and channel to x, y mapping.
        Args:
            model (torch.nn.Module): The neural network model used for predictions.
            space_shape (tuple): The shape of the input space.
            ch2xy (np.ndarray): A mapping from channels to x, y coordinates.
        """
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.space_shape = space_shape
        self.ch2xy = ch2xy
        self.nb_queries = 0  # Initialize the number of queries to zero
        self.mean_queries = 0  # Initialize the mean of the queries to zero
        self.variance_queries = 0  # Initialize the variance of the queries to zero

        # Create a grid of test points in the input space
        self.grid = create_space_test(space_shape)
    
    def from_image_to_vec_for_BO(self, map):
        """
        Converts an image map to a vector for Bayesian Optimization.
        Args:
            map (torch.Tensor): The image map to be converted.
        Returns:
            torch.Tensor: The vector representation of the image map.
        """
        vec = torch.full((len(self.ch2xy),), np.nan)
        for i in range(len(self.ch2xy)):
            vec[i] = map[0, 0, self.ch2xy[i,0]-1, self.ch2xy[i,1]-1]
        return vec

    def get_elem_mean_and_std_fixed_lengthscales(self, kernel_type='Matern52', 
                                    noise=0.05, outputscale=1, lengthscale=[0.3, 0.3]):
        """
        Computes the mean and standard deviation of the GP predictions for all grid points with fixed lengthscales.
        Args:
            kernel_type (str): The type of kernel to use for the GP model.
            noise (float): The noise level for the GP model.
            outputscale (float): The output scale for the GP model.
            lengthscale (list): The lengthscales for the GP model.
        """
        # Initialize tensors to store the mean and standard deviation of the GP predictions
        self.all_maps_elem_mean = torch.full((len(self.grid), 1, 1, self.space_shape[0], self.space_shape[1]), np.nan)
        self.all_maps_elem_std = torch.full((len(self.grid), 1, 1, self.space_shape[0], self.space_shape[1]), np.nan)

        # Loop over each point in the grid
        for i in range(len(self.grid)):
            # Create a Gaussian likelihood and GP model with fixed lengthscales
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            elem_gp = GPytorchFixed(
                            train_x=self.grid[i:i+1], train_y=torch.tensor([1.]), 
                            likelihood=likelihood, kernel_type=kernel_type, noise=noise, 
                            outputscale=outputscale, lengthscale=lengthscale)
            elem_gp.double()
            
            # Predict the mean and standard deviation for the entire grid
            elem_mean, elem_std = elem_gp.predict(self.grid)
            
            # Store the predictions in the corresponding tensors
            for j in range(len(self.grid)):
                k, l = int(self.grid[j,0]*9 + 0.01), int(self.grid[j,1]*9 + 0.01)
                self.all_maps_elem_mean[i, 0, 0, k, l] = elem_mean[j].detach().item()
                self.all_maps_elem_std[i, 0, 0, k, l] = elem_std[j].detach().item()
    
    def get_input_query(self, query_x, query_y, kernel_type='Matern52', 
                                    noise=0.05, outputscale=1, lengthscale=[0.3, 0.3]):
        """
        Gets the input query for the GP model and returns the mean and standard deviation of the predictions.
        Args:
            query_x (torch.Tensor): The input query coordinates.
            query_y (float): The input query value.
            kernel_type (str): The type of kernel to use for the GP model.
            noise (float): The noise level for the GP model.
            outputscale (float): The output scale for the GP model.
            lengthscale (list): The lengthscales for the GP model.
        Returns:
            tuple: The mean and standard deviation of the GP predictions for the input query.
        """
        if hasattr(self, 'all_maps_elem_mean'):
            # If the mean and standard deviation maps are already computed, use them
            k, l = int(query_x[0]*9 + 0.01), int(query_x[1]*9 + 0.01)
            id_grid = k*10 + l
            maps_elem_mean = self.all_maps_elem_mean[id_grid]
            maps_elem_std = self.all_maps_elem_std[id_grid]
            return maps_elem_mean*query_y, maps_elem_std
        else:
            # Otherwise, compute the mean and standard deviation maps
            maps_elem_mean = torch.full((1 , 1, self.space_shape[0], self.space_shape[1]), np.nan)
            maps_elem_std = torch.full((1 , 1, self.space_shape[0], self.space_shape[1]), np.nan)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            elem_gp = GPytorchFixed(
                            train_x=self.idx2coord(query_x), train_y=torch.tensor([1.]), 
                            likelihood=likelihood, kernel_type='Matern52', noise=noise, 
                            outputscale=outputscale, lengthscale=lengthscale)
            elem_gp.double()
            elem_mean, elem_std = elem_gp.predict(self.grid)
            for j in range(len(self.grid)):
                k, l = int(self.grid[j,0]*9 + 0.01), int(self.grid[j,1]*9 + 0.01)
                maps_elem_mean[0, 0, k, l] = elem_mean[j].detach().item()
                maps_elem_std[0, 0, k, l] = elem_std[j].detach().item()
            return maps_elem_mean*query_y, maps_elem_std
    
    def update_with_query(self, query_x, query_y):
        """
        Updates the model with a new query point and its corresponding value.
        Parameters:
        query_x (torch.Tensor): The input query point.
        query_y (float): The observed value at the query point.
        Returns:
        tuple: A tuple containing the updated mean and standard deviation as torch.Tensors.
        """
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