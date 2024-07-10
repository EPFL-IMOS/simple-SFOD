"""
Modules for Monte-Carlo Dropout uncertainty estimation.
"""
import torch.nn as nn


class FCDropoutLayers(nn.Module):

    def __init__(self, in_channels, p):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            p (float): dropout rate
        """
        super(FCDropoutLayers, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, in_channels),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.module(x)
