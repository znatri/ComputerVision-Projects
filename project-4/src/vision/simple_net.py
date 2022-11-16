import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        # super(SimpleNet, self).__init__()
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=500, out_features=100),
            nn.Linear(in_features=100, out_features=15)
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        model_output = self.fc_layers(x)

        ############################################################################
        # Student code end
        ############################################################################

        return model_output
