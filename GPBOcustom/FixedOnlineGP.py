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

def schur_inverse(Dinv: np.ndarray, B: np.ndarray, A: float):
    """
    Compute the inverse of a block matrix K using Schur complement.
    
    K = [[D, B.T],
         [B, A  ]]
    
    Parameters:
        Dinv (np.array): The inverse of the top-left matrix D.
        B (np.array): A row vector (top-right of the block matrix).
        A (float): A scalar (bottom-right of the block matrix).

    Returns:
        Kinv (np.array): The inverse of the block matrix K.
    """
    
    # Step 1: Compute the Schur complement of D in K
    # S = A - B @ Dinv @ B.T
    Schur_complement = A - B @ Dinv @ B.T
    
    # Step 2: Compute the inverse of the Schur complement (since it's scalar, we just invert it)
    Schur_complement_inv = 1.0 / Schur_complement
    
    # Step 3: Compute the blocks of the inverse matrix Kinv
    # Top-left block: Dinv + Dinv @ B.T @ (A - B @ Dinv @ B.T)^-1 @ B @ Dinv
    K11_inv = Dinv + Dinv @ B.T * Schur_complement_inv @ B @ Dinv
    
    # Top-right block: - Dinv @ B.T @ (A - B @ Dinv @ B.T)^-1
    K12_inv = - Dinv @ B.T * Schur_complement_inv
    
    # Bottom-left block: same as K12_inv.T because of symmetry
    K21_inv = K12_inv.T
    
    # Bottom-right block: (A - B @ Dinv @ B.T)^-1
    K22_inv = Schur_complement_inv

    # Step 4: Combine the blocks to form the inverse matrix Kinv
    # Kinv = [[K11_inv, K12_inv],
    #         [K21_inv, K22_inv]]
    Kinv = np.block([[K11_inv, K12_inv],
                     [K21_inv, K22_inv]])
    
    return Kinv

class FixedOnlineGP:

    def __init__(self, input_space: np.ndarray, kernel_type: str = 'rbf', noise_std: float = 0.1, output_std: float = 1, lengthscale: float = 0.05, NB_IT: int = None) -> None:
        self.input_space = input_space  # Input space

        self.kernel_type = kernel_type  # Kernel type
        self.noise_std = noise_std  # Standard deviation of the noise
        self.output_std = output_std  # Standard deviation of the output
        self.lengthscale = lengthscale  # Lengthscale parameter for the kernel

        self.space_size = input_space.shape[0]  # Number of queries in the input space
        self.space_dim = input_space.shape[1]  # Dimensionality of the input space

        # Initialize mean and standard deviation arrays
        self.mean = np.zeros(self.space_size)
        self.std = np.zeros(self.space_size)

        if NB_IT is None: # if NB_IT not specified 
            self.NB_IT = self.space_size   
        else:
            self.NB_IT = NB_IT
        self.nb_queries = 0  # Number of training samples
        
    def set_kernel(self) -> None:
        """
        set the kernel

        Raises:
            ValueError: If the kernel_type is not recognized.
        """
        if self.kernel_type == 'rbf':
            self.kernel = GPy.kern.RBF(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale)
        elif self.kernel_type == 'Mat32':
            self.kernel = GPy.kern.Matern32(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale)
        elif self.kernel_type == 'Mat52':
            self.kernel = GPy.kern.Matern52(input_dim=self.space_dim, variance=self.output_std**2, lengthscale=self.lengthscale)
        else:
            raise ValueError("The attribute kernel_type is not well defined")
        
    def update_no_schur(self, query_x: np.ndarray, query_y: float) -> None:
        """_summary_

        Args:
            query_x (np.ndarray): _description_
            query_y (float): _description_
        """
        if self.nb_queries == 0:

            self.queries_X = np.zeros((self.NB_IT, self.space_dim))
            self.queries_Y = np.zeros((self.NB_IT, 1))
            self.kernel_vect_mat = np.zeros((self.space_size, self.NB_IT))

            self.queries_X[self.nb_queries, :] = query_x  # query's shape is (space_dim)
            self.queries_Y[self.nb_queries, 0] = query_y  # query_y is float

            self.kernel_mat = np.array([[self.output_std**2]])

            self.K_inv = np.array([[1.0 / (self.output_std**2 + self.noise_std**2)]])

            self.kernel_vect_mat[:, self.nb_queries] = self.kernel.K(self.input_space,
                                                                     self.queries_X[self.nb_queries:self.nb_queries+1, :])[:,0] # Covariance vector between input_space et query shape(space_size,1)
        
        else:
            self.queries_X[self.nb_queries, :] = query_x  # query's shape is (space_dim)
            self.queries_Y[self.nb_queries, 0] = query_y  # query_y is float

            vect = self.kernel.K(self.queries_X[self.nb_queries:self.nb_queries+1, :], self.queries_X[:self.nb_queries, :])      # shape(1, nb_query)

            self.kernel_mat = np.block([[self.kernel_mat, vect.T          ],       
                                        [vect           , self.output_std**2]])    

            # Add noise to the kernel matrix
            K = self.kernel_mat + self.noise_std**2 * np.eye(self.nb_queries+1)

            # Perform Cholesky decomposition
            c, low = cho_factor(K)  # Returns the Cholesky decomposition of the matrix K
        
            # Solve for the inverse of the matrix K using the Cholesky factor
            self.K_inv = cho_solve((c, low), np.eye(K.shape[0]))  # Inverse the matrix

            self.kernel_vect_mat[:, self.nb_queries] = self.kernel.K(self.input_space, 
                                                                     self.queries_X[self.nb_queries:self.nb_queries+1, :])[:,0] # Covariance vector between input_space et query shape(space_size,1)

        self.nb_queries += 1

    def update(self, query_x: np.ndarray, query_y: float) -> None:
        """_summary_

        Args:
            query_x (np.ndarray): _description_
            query_y (float): _description_
        """
        if self.nb_queries == 0:

            self.queries_X = np.zeros((self.NB_IT, self.space_dim))
            self.queries_Y = np.zeros((self.NB_IT, 1))
            self.kernel_vect_mat = np.zeros((self.space_size, self.NB_IT))

            self.queries_X[self.nb_queries, :] = query_x  # query's shape is (space_dim)
            self.queries_Y[self.nb_queries, 0] = query_y  # query_y is float

            self.kernel_mat = np.array([[self.output_std**2]])

            self.K_inv = np.array([[1.0 / (self.output_std**2 + self.noise_std**2)]])

            self.kernel_vect_mat[:, self.nb_queries] = self.kernel.K(self.input_space,
                                                                     self.queries_X[self.nb_queries:self.nb_queries+1, :])[:,0] # Covariance vector between input_space et query shape(space_size,1)
        
        else:
            self.queries_X[self.nb_queries, :] = query_x  # query's shape is (space_dim)
            self.queries_Y[self.nb_queries, 0] = query_y  # query_y is float

            vect = self.kernel.K(self.queries_X[self.nb_queries:self.nb_queries+1, :], self.queries_X[:self.nb_queries, :])      # shape(1, nb_query)

            self.kernel_mat = np.block([[self.kernel_mat, vect.T          ],       
                                        [vect           , self.output_std**2]])    

            self.K_inv = schur_inverse(A = self.output_std**2 + self.noise_std**2, B = vect, Dinv = self.K_inv)

            self.kernel_vect_mat[:, self.nb_queries] = self.kernel.K(self.input_space, 
                                                                     self.queries_X[self.nb_queries:self.nb_queries+1, :])[:,0] # Covariance vector between input_space et query shape(space_size,1)

        self.nb_queries += 1

    def predict(self) -> None:
        
        # Compute the predicted mean and standard deviation for each point in the input space
        for i in range(self.space_size):
            kernel_vect = self.kernel_vect_mat[i, :self.nb_queries]  # Covariance vector for the current input space point
            self.mean[i] = kernel_vect @ self.K_inv @ self.queries_Y[:self.nb_queries, :]  # Compute the mean
            self.std[i] = np.sqrt(self.output_std**2 
                                  - kernel_vect @ self.K_inv @ kernel_vect)  # Compute the standard deviation