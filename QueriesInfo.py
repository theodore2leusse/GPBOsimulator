# sciNeurotech Lab 
# Theodore

"""
This file contains the QueriesInfo class, which is used to manage and update query information for Bayesian optimization.
"""

import numpy as np
import torch
import gpytorch
from GPcustom.models import GPytorchModel
from botorch.utils.transforms import standardize

def matern52_kernel(i, j, shape, lengthscale):
    """
    Generates a Matern52 kernel centered at (i, j) for a matrix of size `shape`.

    Args:
        i, j (int): Coordinates of the kernel's center.
        shape (tuple): Size of the matrix (e.g., (10, 10)).
        lengthscale (list): Controls the kernel's range in x and y directions [lx, ly].
                            Allows for anisotropic kernels (e.g., elliptical shapes).

    Returns:
        np.ndarray: A matrix representing the Matern52 kernel, normalized to have a maximum value of 1 at the center.
    """
    # Create coordinate grids for x and y
    x, y = np.indices(shape)
    
    # Compute normalized distances from the center (i, j)
    dx = (x - i) / (lengthscale[0] * (shape[0]-1))  # Normalized distance in the x-direction
    dy = (y - j) / (lengthscale[1] * (shape[1]-1))  # Normalized distance in the y-direction
    r = np.sqrt(dx**2 + dy**2)     # Euclidean distance (anisotropic if lx != ly)
    
    # Matern52 kernel formula
    kernel = (1 + np.sqrt(5) * r + (5 / 3) * r**2) * np.exp(-np.sqrt(5) * r)
    
    # Normalize the kernel so that its maximum value is 1 at the center (i, j)
    kernel /= kernel.max()
    
    return kernel

def apply_matern52_kernels(counts_array, lengthscale):
    """
    Applies Matern52 kernels to a 2D array based on integer counts at each coordinate.
    
    For each cell (i, j) in `counts_array`, adds `N` Matern52 kernels centered at (i, j),
    where `N` is the integer value of `counts_array[i, j]`.
    
    Args:
        counts_array (np.ndarray): 2D array of integers. Each value represents the number
                                   of Matern52 kernels to apply at its coordinates.
        lengthscale (list): Lengthscale parameters [lx, ly] for the Matern52 kernel.
    
    Returns:
        np.ndarray: Resulting 2D array after summing all Matern52 kernels.
    """
    # Initialize the result matrix with zeros (float type for smooth gradients)
    result = np.zeros_like(counts_array, dtype=np.float64)
    
    # Get the shape of the input array
    rows, cols = counts_array.shape
    
    # Iterate over every cell in the counts_array
    for i in range(rows):
        for j in range(cols):
            # Get the number of kernels to apply at this position
            num_kernels = counts_array[i, j]
            
            if num_kernels > 0:
                # Generate the Matern52 kernel centered at (i, j)
                kernel = matern52_kernel(
                    i=i, 
                    j=j, 
                    shape=counts_array.shape, 
                    lengthscale=lengthscale
                )
                
                # Add the kernel multiplied by the count to the result
                result += num_kernels * kernel
                
    return result

import numpy as np
import torch
import gpytorch

class QueriesInfo:
    """
    A class to manage and update query information for Bayesian optimization. 
    Also allows us to make Sparse Variational Gaussian Process (cf. methods estimated_gpytorch & estimated_hp_gpytorch in SimGPBO.py).

    Attributes:
        space_shape (tuple): The shape of the search space.
        query_map (np.ndarray): A map to keep track of the number of queries at each point in the search space.
        mean_map (np.ndarray): A map to store the mean of the query results at each point in the search space.
        var_map (np.ndarray): A map to store the variance of the query results at each point in the search space.
    """

    def __init__(self, space_shape):
        """
        Initializes the QueriesInfo object.

        Args:
            space_shape (tuple): The shape of the search space.
        """
        self.space_shape = space_shape
        self.query_map = np.full(space_shape, 0)
        self.mean_map = np.full(space_shape, np.nan)
        self.var_map = np.full(space_shape, np.nan)

    def update_map(self, query_x, query_y):
        """
        Updates the query map, mean map, and variance map with new query results.

        Args:
            query_x (tuple): The coordinates of the query in the search space.
            query_y (float): The result of the query.
        """
        self.query_map[query_x] += 1
        if self.query_map[query_x] == 1:
            self.mean_map[query_x] = query_y
        elif self.query_map[query_x] == 2:
            new_mean_map = (query_y + self.mean_map[query_x]) / 2
            self.var_map[query_x] = ((query_y - new_mean_map) ** 2 + (self.mean_map[query_x] - new_mean_map) ** 2) / 2
            self.mean_map[query_x] = new_mean_map
        else:
            new_mean_map = ((self.query_map[query_x] - 1) * self.mean_map[query_x] + query_y) / self.query_map[query_x]
            self.var_map[query_x] = ((self.query_map[query_x] - 1) * (self.var_map[query_x] + (new_mean_map - self.mean_map[query_x]) ** 2) + (query_y - self.mean_map[query_x]) ** 2) / self.query_map[query_x]
            self.mean_map[query_x] = new_mean_map

    def mat2vec(self, mat, ch2xy):
        """
        Converts a matrix to a vector based on a given mapping.

        Args:
            mat (np.ndarray): The input matrix.
            ch2xy (list): A list of coordinates to extract from the matrix.

        Returns:
            np.ndarray: A vector containing the values from the matrix at the specified coordinates.
        """
        vec = np.zeros(len(ch2xy))
        for i in range(len(ch2xy)):
            vec[i] = mat[*(ch2xy[i] - 1)]
        return vec

    def idx2coord(self, idx):
        """
        Converts indices to normalized coordinates.

        Args:
            idx (tuple): The indices to convert.

        Returns:
            tuple: The normalized coordinates.
        """
        coord = []
        for i in range(len(idx)):
            coord.append(idx[i] / (self.space_shape[i] - 1))
        return tuple(coord)

    def coord2idx(self, coord):
        """
        Converts normalized coordinates to indices.

        Args:
            coord (tuple): The normalized coordinates to convert.

        Returns:
            tuple: The indices.
        """
        idx = []
        for i in range(len(coord)):
            idx.append(round(coord[i] * (self.space_shape[i] - 1)))
        return tuple(idx)

    def get_mean_queries(self):
        """
        Retrieves a list of mean queries and their coordinates.

        Returns:
            list: A list of tuples containing the coordinates and mean query values.
        """
        mean_queries_list = []
        for idx in np.ndindex(self.space_shape):
            if not np.isnan(self.mean_map[idx]):
                mean_queries_list.append([self.idx2coord(idx), self.mean_map[idx]])
        return mean_queries_list

    def estimate_HP(self, outputscale: float = None, noise: float = None, max_iters_training_gp: int = 100, lr: float = 0.1):
        """
        Estimates the hyperparameters of the Gaussian Process model.

        Args:
            outputscale (float, optional): The output scale for the GP model. Defaults to None.
            noise (float, optional): The noise level for the GP model. Defaults to None.
            max_iters_training_gp (int, optional): The maximum number of iterations for training the GP model. Defaults to 100.
            lr (float, optional): The learning rate for training the GP model. Defaults to 0.1.
        """
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
        self.gp.train_model(train_x, train_y, max_iters=max_iters_training_gp, lr=lr, Verbose=False)
        self.hyperparams = self.gp.get_hyperparameters()

    def predict(self, test_X, ch2xy, alpha: float = 1):
        """
        Predicts the mean and standard deviation for a given set of test points.

        Args:
            test_X (torch.Tensor): The test points for prediction.
            ch2xy (list): A list of coordinates to extract from the query map.
            alpha (float, optional): A scaling factor for the blurring kernel. Defaults to 1.

        Returns:
            tuple: The predicted mean and standard deviation.
        """
        if alpha > 0:
            blurred_query_map = apply_matern52_kernels(self.query_map, [x * alpha for x in self.hyperparams['lengthscale']])
        else:
            blurred_query_map = self.query_map
        gp_mean_pred, gp_std_pred = self.gp.predict(test_X)
        gp_std_pred = gp_std_pred / (self.mat2vec(blurred_query_map, ch2xy) + 1)

        return gp_mean_pred, gp_std_pred

        
