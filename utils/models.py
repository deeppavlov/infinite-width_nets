import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, input_shape, num_classes, width, num_hidden=1, bias=True):
        super(FCNet, self).__init__()
        
        input_size = int(np.prod(input_shape))
        
        self.input_layer = nn.Linear(input_size, width, bias=bias)
        
        hidden_layers = []
        hidden_layers.append(nn.ReLU())
        
        for _ in range(num_hidden-1):
            hidden_layers.append(nn.Linear(width, width, bias=bias))
            hidden_layers.append(nn.ReLU())
            
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.output_layer = nn.Linear(width, num_classes, bias=bias)
        
    def forward(self, X):
        Z = self.input_layer(X.view(X.shape[0], -1))
        Z = self.hidden_layers(Z)
        return self.output_layer(Z.view(Z.shape[0], -1))