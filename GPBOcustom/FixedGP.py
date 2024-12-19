# sciNeurotech Lab 
# Theodore

"""
This module defines a class for a Gaussian Process (GP) with fixed lengthscales.
The GP can use different kernel types for function approximation with noise.
"""

# Import necessary libraries
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import GPy

class FixedGP:
    """
    A class representing a Gaussian Process with fixed hyperparameters.

    Methods:
        compute_kernel: Computes the covariance matrix and the kernel vector matrix based on the selected kernel type.
        predict: Predicts the mean and standard deviation of the GP at the given input space.

    Attributes:
        input_space (np.ndarray): The input space for the GP. 
        train_X (np.ndarray): The training input data.
        train_Y (np.ndarray): The training output data.
        kernel_type (str): The type of kernel to be used ('rbf', 'Mat32', or 'Mat52').
        noise_std (float): The standard deviation of the noise in the output.
        output_std (float): The standard deviation of the output function.
        lengthscale (float): The lengthscale parameter for the kernel.
        space_size (int): The number of queries in the input space.
        nb_queries (int): The number of training samples.
        space_dim (int): The dimensionality of the input space.
        kernel (GPy.kern): The kernel object used in the GP.
        kernel_mat (np.ndarray): The covariance matrix of the training inputs.
        kernel_vect_mat (np.ndarray): The covariance matrix between input space and training inputs.
        K_inv (np.ndarray): The inverse of the kernel matrix.
        mean (np.ndarray): The predicted mean values at the input space.
        std (np.ndarray): The predicted standard deviations at the input space.

    Methods:
        set_kernel: Sets the kernel based on the specified type and hyperparameters.
        compute_kernel: Computes the covariance matrices for the GP model.
        predict: Predicts the mean and standard deviation of the GP for the input space.
    """

    def __init__(self, input_space: np.ndarray, train_X: np.ndarray, train_Y: np.ndarray, kernel_type: str = 'rbf', noise_std=0.1, output_std=1, lengthscale=0.05) -> None:
        """
        Initializes the FixedLengthscalesGP instance.

        Args:
            input_space (np.ndarray): The input space for the GP. shape(space_size, space_dim)
            train_X (np.ndarray): The training input data. shape(train_size, space_dim)
            train_Y (np.ndarray): The training output data. shape(train_size, 1)
            kernel_type (str, optional): The type of kernel to be used. Defaults to 'rbf'.
            noise_std (float, optional): The standard deviation of the noise in the output. Defaults to 0.1.
            output_std (float, optional): The standard deviation of the output function. Defaults to 1.
            lengthscale (float, optional): The lengthscale parameter for the kernel. Defaults to 0.05.
        """
        self.input_space = input_space  # Input space
        self.train_X = train_X  # Training input data
        self.train_Y = train_Y  # Training output data

        self.kernel_type = kernel_type  # Kernel type
        self.noise_std = noise_std  # Standard deviation of the noise
        self.output_std = output_std  # Standard deviation of the output
        self.lengthscale = lengthscale  # Lengthscale parameter for the kernel

        self.space_size = input_space.shape[0]  # Number of queries in the input space
        self.space_dim = input_space.shape[1]  # Dimensionality of the input space
        self.nb_queries = train_X.shape[0]  # Number of training samples

    def set_kernel(self) -> None:
        """
        Sets the kernel object based on the kernel type and hyperparameters.

        Raises:
            ValueError: If `kernel_type` is not recognized or `lengthscale` is invalid.
        """
        if isinstance(self.lengthscale, float) or (isinstance(self.lengthscale, list) and len(self.lengthscale) == 1):     
            if self.kernel_type == 'rbf':
                self.kernel = GPy.kern.RBF(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale)
            elif self.kernel_type == 'Mat32':
                self.kernel = GPy.kern.Matern32(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale)
            elif self.kernel_type == 'Mat52':
                self.kernel = GPy.kern.Matern52(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale)
            else:
                raise ValueError("The attribute kernel_type is not well defined")
        elif len(self.lengthscale) == self.space_dim:
            if self.kernel_type == 'rbf':
                self.kernel = GPy.kern.RBF(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale, ARD=True)
            elif self.kernel_type == 'Mat32':
                self.kernel = GPy.kern.Matern32(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale, ARD=True)
            elif self.kernel_type == 'Mat52':
                self.kernel = GPy.kern.Matern52(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale, ARD=True)
            else:
                raise ValueError("The attribute kernel_type is not well defined")
        else:
            raise ValueError("The attribute lengthscale is not well defined")

    def compute_kernel(self) -> None:
        """
        Computes the covariance matrix and the kernel vector matrix based on the selected kernel type.
        """
        self.kernel_mat = self.kernel.K(self.train_X)  # Covariance matrix of the training inputs
        self.kernel_vect_mat = self.kernel.K(self.input_space, self.train_X)  # Covariance matrix between input space and training inputs. shape = (space_size, train_size)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the mean and standard deviation of the GP at the given input space.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - mean (np.ndarray): Predicted mean values for the input space (shape: (space_size,)).
                - std (np.ndarray): Predicted standard deviations for the input space (shape: (space_size,)).
        """
        self.set_kernel()
        self.compute_kernel()  # Compute kernel matrices
        
        # Add noise to the kernel matrix
        K = self.kernel_mat + self.noise_std**2 * np.eye(self.nb_queries)

        # Perform Cholesky decomposition
        c, low = cho_factor(K)  # Returns the Cholesky decomposition of the matrix K
        
        # Solve for the inverse of the matrix K using the Cholesky factor
        self.K_inv = cho_solve((c, low), np.eye(K.shape[0]))  # Inverse the matrix
      
        # Compute all mean values in one operation
        self.mean = self.kernel_vect_mat @ (self.K_inv @ self.train_Y[:,0])

        # Compute the std for each point in a vectorized manner
        kernel_diag = np.einsum('ij,ji->i', self.kernel_vect_mat, self.K_inv @ self.kernel_vect_mat.T)
        if max(kernel_diag) > self.output_std**2:
            print('we have a problem, we have a negative variance')
        self.std = np.sqrt(self.output_std**2 - kernel_diag + self.noise_std**2)

        return self.mean.copy(), self.std.copy()


