# sciNeurotech Lab 
# Theodore

import torch 
import torch.nn as nn
import numpy as np 
import math
import matplotlib.pyplot as plt 

"""
This module contains utility functions and classes for using the NNs in the project.

For example, this file contains everything needed to use a neural network with architecture 7. (see simuNN.ipynb & simGPBO.py for usage examples)
"""

class MapUpdateNetwork_7(nn.Module):
    """
    A convolutional neural network designed for processing and updating map-like structures.
    The network takes four input images as input, encodes their features, and decodes them 
    to generate an output map of the same spatial dimensions as the inputs.

    Attributes:
        encoder (nn.Sequential): Encoder that reduces spatial dimensions and extracts high-level features.
        decoder (nn.Sequential): Decoder that reconstructs the spatial dimensions from encoded features.

    Methods:
        __init__(in_channels, out_channels, out_channels_first_conv):
            Initializes the network with configurable input/output channels and feature depth.
        forward(img1, img2, img3, img4):
            Performs the forward pass by concatenating the input images, encoding their features, 
            and decoding them to produce the final map.
    """

    def __init__(self, in_channels=4, out_channels=2, out_channels_first_conv=16, std_max=math.sqrt(1+0.05)):
        """
        Initializes the MapUpdateNetwork.

        Args:
            in_channels (int): Number of input channels. Defaults to 4 (for concatenated input images).
            out_channels (int): Number of output channels. Defaults to 2.
            out_channels_first_conv (int): Number of channels in the first convolutional layer. Defaults to 16.
            std_max(float, obtional): correspond à la valeur maximale atteignable par std. Pour la calculer, il existe la formule suivante :
                                                        std_max = sqrt(outputscale+noise)
                                      Default to sqrt(1+0.05)
        """
        super(MapUpdateNetwork_7, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_first_conv, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(out_channels_first_conv, 2*out_channels_first_conv, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(2*out_channels_first_conv, 4*out_channels_first_conv, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(4*out_channels_first_conv, 8*out_channels_first_conv, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(8*out_channels_first_conv, 16*out_channels_first_conv, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16*out_channels_first_conv, 8*out_channels_first_conv, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8*out_channels_first_conv, 4*out_channels_first_conv, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4*out_channels_first_conv, 2*out_channels_first_conv, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*out_channels_first_conv, out_channels_first_conv, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        self.final_layer1 = nn.Conv2d(out_channels_first_conv, 1, kernel_size=3, stride=1, padding=1) 
        self.final_layer2 = nn.Sequential(
            nn.Conv2d(out_channels_first_conv, 1, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid()
        )

        self.train_losses = []
        self.validation_losses = []
        self.std_max = std_max

    def forward(self, img1, img2, img3, img4):
        """
        Forward pass of the network. Combines four input images, processes them through
        an encoder-decoder architecture, and outputs an updated map.

        Args:
            img1 (torch.Tensor): First input image tensor of shape (B, C, H, W).
            img2 (torch.Tensor): Second input image tensor of shape (B, C, H, W).
            img3 (torch.Tensor): Third input image tensor of shape (B, C, H, W).
            img4 (torch.Tensor): Fourth input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output map tensor of shape (B, out_channels, H, W), where 
                          H and W are the height and width of the input images.
        """
        x = torch.cat((img1, img2, img3, img4), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        channel1 = self.final_layer1(x)  # Output non contraint
        channel2 = self.std_max * self.final_layer2(x)  # Contraint entre [0, 3]
        return torch.cat((channel1, channel2), dim=1)
    
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()  # Si les valeurs dans [0, 3] représentent des classes discrètes

def custom_loss(output, targets):
    target1 = targets[:, 0]  # Extraire les cibles pour le premier canal
    target2 = targets[:, 1]  # Extraire les cibles pour le deuxième canal
    loss1 = criterion1(output[:, 0], target1)
    loss2 = criterion2(output[:, 1], target2) 
    return loss1 + loss2

def evaluate_model(model, test_loader, criterion_function, device="cpu"):
    """
    Evaluates the trained model on a test dataset.

    Args:
        model (nn.Module): Trained neural network.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion_function : Loss function.
        device (str): Device to use ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        float: Average loss on the test set.
    """
    model.eval()
    model.to(device)
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(*torch.split(inputs, 1, dim=1))
            loss = criterion_function(outputs, targets)
            test_loss += loss.item()
    return test_loss / len(test_loader)

def load_my_model(model_path, model_t, optimizer_t=None, scheduler_t=None, plot=False):
    """
    Loads a model from a checkpoint file, along with its optimizer and scheduler states if provided.
    Optionally evaluates the model on a test set and plots the training and validation losses.

    Args:
        model_path (str): Path to the checkpoint file.
        model_t (nn.Module): Model instance to load the weights into.
        optimizer_t (torch.optim.Optimizer, optional): Optimizer instance to load the state into. Defaults to None.
        scheduler_t (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler instance to load the state into. Defaults to None.
        plot (bool, optional): Whether to plot the training and validation losses. Defaults to False.

    Returns:
        Depending on the provided arguments, returns the model, optimizer, scheduler, and losses.
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the checkpoint file
    if device == "cpu":
        checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path, weights_only=True)

    # Load model weights
    model_t.load_state_dict(checkpoint['model_state_dict'])

    if optimizer_t is not None:
        # Load optimizer state
        optimizer_t.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler_t is not None:
        # Load scheduler state
        scheduler_t.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load training and validation losses
    train_losses_t = checkpoint['train losses']
    validation_losses_t = checkpoint['validation losses']
    
    if plot:
        # Plot the training and validation losses
        plt.plot(np.arange(len(train_losses_t)) + 1, train_losses_t, label='train losses')
        plt.plot(np.arange(len(validation_losses_t)) + 1, validation_losses_t, label='validation losses')
        plt.legend()
        plt.xlabel('Number of epochs')
        plt.ylabel('MSE loss (log scale)')
        plt.yscale('log')
        plt.title("Evolution of losses during training")
        plt.show()

    # Return the appropriate objects based on the provided arguments
    if (optimizer_t is None) and (scheduler_t is None):
        return model_t, train_losses_t, validation_losses_t
    elif (optimizer_t is not None) and (scheduler_t is None):
        return model_t, optimizer_t, train_losses_t, validation_losses_t
    elif (optimizer_t is None) and (scheduler_t is not None):
        return model_t, scheduler_t, train_losses_t, validation_losses_t
    else:
        return model_t, optimizer_t, scheduler_t, train_losses_t, validation_losses_t