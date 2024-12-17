import torch
import gpytorch
import botorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize, normalize

# Define a class for a custom GPytorch model
class BOtorchModel(botorch.models.SingleTaskGP):
    def __init__(self, train_X, train_Y, kernel_type: str = 'Matern52'):
        """
        Initializes the Gaussian Process model with the specified kernel.

        Args:
            train_X (torch.Tensor): Input training data.
            train_Y (torch.Tensor): Output training data.
            kernel_type (str): The type of kernel to use ('Matern52' or 'RBF').
        """
        # Choose the kernel based on the 'kernel_type' argument
        if kernel_type == 'Matern52':
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_X.shape[1])
            )
        elif kernel_type == 'RBF':
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_X.shape[1])
            )
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        # Call the parent constructor 
        super().__init__(train_X, train_Y, covar_module=covar_module)
        # super().__init__(train_X, train_Y, covar_module=gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_X.shape[1]))

        # Initialize Gaussian likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Save training data for potential reuse
        self.train_X = train_X
        self.train_Y = train_Y

    def fit_model(self):
        """
        Fits the model to the training data and returns the marginal log-likelihood (MLL).
        """
        # Switch to training mode
        self.train()
        self.likelihood.train()

        # Create the exact marginal log-likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        # Optimize hyperparameters using BoTorch
        fit_gpytorch_mll(mll)

        # Switch to training mode
        self.train()
        self.likelihood.train()

        # Forward pass to compute the MLL on training data
        output = self(self.train_X)
        loss = -mll(output, self.train_Y[:, 0])  # Compute the negative MLL

        return loss.item()

    def get_hyperparameters(self):
        """
        Retrieves the model's hyperparameters.

        Returns:
            dict: A dictionary containing the lengthscale, outputscale, and noise level.
        """
        return {
            'lengthscale': self.covar_module.base_kernel.lengthscale[0].detach().tolist(),
            'outputscale': self.covar_module.outputscale.item(),
            'noise': self.likelihood.noise.item()
        }

    def predict(self, test_X):
        """
        Makes predictions on new data.

        Args:
            test_X (torch.Tensor): Input test data.

        Returns:
            tuple: Mean and standard deviation of predictions.
        """
        # Switch to evaluation mode
        self.eval()
        self.likelihood.eval()

        # Perform predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(test_X))
            self.mean = pred.mean
            self.std = pred.stddev

        return self.mean.clone(), self.std.clone()

# Example usage
if __name__ == "__main__":
    # Generate example training data
    train_X = torch.rand(10, dtype=torch.float64).view(-1, 1)
    train_Y = torch.sin(train_X * (2 * torch.pi)) + 0.2 * torch.randn_like(train_X)

    # Standardize the output
    train_Y = standardize(train_Y)

    # Instantiate the model
    model = BOtorchModel(train_X, train_Y)

    # Convert model to double precision
    model.double()

    # Fit the model
    loss = model.fit_model()
    print(f"Training Loss (Negative Marginal Log-Likelihood): {loss:.3f}")

    # Retrieve hyperparameters
    print(model.get_hyperparameters())

    # Generate test points
    test_X = torch.linspace(0, 1, 200, dtype=torch.float64).view(-1, 1)

    # Make predictions
    mean, std = model.predict(test_X)

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(train_X.numpy(), train_Y.numpy(), 'k*', label='Training Data')
    plt.plot(test_X.numpy(), mean.numpy(), 'b', label='Mean Prediction')
    plt.fill_between(
        test_X.squeeze().numpy(),
        mean.numpy() - 1.96 * std.numpy(),
        mean.numpy() + 1.96 * std.numpy(),
        color='blue',
        alpha=0.2,
        label='Confidence Interval'
    )
    plt.legend()
    plt.show()


