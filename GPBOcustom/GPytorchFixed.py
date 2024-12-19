import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
import matplotlib.pyplot as plt

class GPytorchFixed(gpytorch.models.ExactGP):
    """
    A Gaussian Process model using GPytorch with fixed hyperparameters.
    
    Attributes:
        train_x (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training target data.
        likelihood (gpytorch.likelihoods): Likelihood function.
        kernel_type (str): Type of kernel ('Matern52' or 'RBF').
        lengthscale (float or torch.Tensor): Fixed lengthscale parameter.
        outputscale (float): Fixed output scale parameter.
        noise (float): Fixed noise parameter.
    """

    def __init__(self, train_x, train_y, likelihood, kernel_type: str = 'Matern52',
                 lengthscale=None, outputscale=None, noise=None):
        """
        Initializes the Gaussian Process model with fixed hyperparameters.

        Args:
            train_x (torch.Tensor): Input training data.
            train_y (torch.Tensor): Target training data.
            likelihood (gpytorch.likelihoods): Gaussian likelihood function.
            kernel_type (str): Kernel type, either 'Matern52' or 'RBF'.
            lengthscale (float or torch.Tensor): Fixed lengthscale parameter.
            outputscale (float): Fixed output scale parameter.
            noise (float): Fixed noise parameter.
        """
        super().__init__(train_x, train_y, likelihood)
        
        # Define kernel based on 'kernel_type' parameter
        if kernel_type == 'Matern52':
            kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        elif kernel_type == 'RBF':
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        self.mean_module = gpytorch.means.ZeroMean()  # Default mean function is ConstantMean() and not ZeroMean(), ConstantMean() add a hyperparameter to estimate
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

        # Set fixed hyperparameters if provided
        if lengthscale is not None:
            if isinstance(lengthscale, float):
                lengthscale = torch.tensor([lengthscale] * train_x.shape[1], dtype=torch.float64)
            self.covar_module.base_kernel.lengthscale = lengthscale
            self.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        if outputscale is not None:
            self.covar_module.outputscale = outputscale
            self.covar_module.raw_outputscale.requires_grad = False
            
        if noise is not None:
            self.likelihood.noise = noise
            self.likelihood.raw_noise.requires_grad = False

    def forward(self, x):
        """
        Forward pass for the GP model.

        Args:
            x (torch.Tensor): Input data for prediction.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predictive distribution.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def get_hyperparameters(self):
        """
        Get the current hyperparameters of the model.
        
        Returns:
            dict: Dictionary containing the current hyperparameters
        """
        return {
            'lengthscale': self.covar_module.base_kernel.lengthscale[0].detach().tolist(),
            'outputscale': self.covar_module.outputscale.detach().item(),
            'noise': self.likelihood.noise.detach().item(),
        }

    def predict(self, test_X):
        """
        Makes predictions on new data.

        Args:
            test_X (torch.Tensor): Input test data.

        Returns:
            tuple: Mean and standard deviation of predictions.
        """ 
        self.eval()
        self.likelihood.eval()

        if test_X.dtype != torch.float64:
            test_X = test_X.double()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self(test_X)
            pred = self.likelihood(output)
        
        mean = pred.mean
        std = pred.variance.sqrt()

        return mean.clone(), std.clone()

if __name__ == "__main__":
    # Example usage
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Generate example training data
    train_X = torch.linspace(0, 1, 15, dtype=torch.float64).view(-1, 1)
    train_Y = (torch.sin(train_X * (2 * torch.pi)) + 0.1 * torch.randn_like(train_X))[:, 0]
    train_Y = standardize(train_Y)

    # Fixed hyperparameters
    fixed_params = {
        'lengthscale': 0.2,
        'outputscale': 1.0,
        'noise': 0.1
    }

    # Instantiate the fixed GP model
    model = GPytorchFixed(
        train_X, train_Y, likelihood, 
        kernel_type='Matern52',
        **fixed_params
    )

    print(model.get_hyperparameters())

    # Generate test points
    test_X = torch.linspace(0, 1, 200, dtype=torch.float64).view(-1, 1)

    # Make predictions
    pred_mean, pred_std = model.predict(test_X)

    # Plotting results
    plt.figure(figsize=(8, 6))
    plt.plot(train_X.numpy(), train_Y.numpy(), 'k*', label='Training Data')
    plt.plot(test_X.numpy(), pred_mean.numpy(), 'b', label='Mean Prediction')
    plt.fill_between(
        test_X.squeeze().numpy(),
        (pred_mean - 1.96 * pred_std).numpy(),
        (pred_mean + 1.96 * pred_std).numpy(),
        color='blue', alpha=0.2, label='Confidence Interval'
    )
    plt.legend()
    plt.title("Fixed Gaussian Process Regression")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show() 