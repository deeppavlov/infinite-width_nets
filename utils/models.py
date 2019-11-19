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
    
    
def make_layers(config, input_shape, init_num_kernels, use_bn, bias, dropout_rate):
    out_channels, h, w = input_shape
    num_kernels = init_num_kernels
    layers = []
    for layer_name in config:
        if layer_name == 'conv':
            in_channels, out_channels = out_channels, num_kernels
            layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias), nn.LeakyReLU(0.1)]
            if use_bn:
                layers += [nn.BatchNorm2d(out_channels)]
        elif layer_name == 'max-pool':
            layers += [nn.MaxPool2d(2), nn.Dropout2d(dropout_rate)]
            num_kernels *= 2
            h = (h+1) // 2
            w = (w+1) // 2
        elif layer_name == 'avg-pool':
            layers += [nn.AvgPool2d((h,w)), nn.Flatten()]
        elif layer_name == 'dense':
            num_kernels //= 2
            in_channels, out_channels = out_channels, num_kernels
            layers += [nn.Linear(in_channels, out_channels, bias=bias), nn.LeakyReLU(0.1)]
            if use_bn:
                layers += [nn.BatchNorm1d(out_channels, affine=False)]
                
    return layers, out_channels

    
class ConvLarge(nn.Module):
    def __init__(self, input_shape, num_classes, init_num_kernels, use_bn=True, hidden_bias=False, output_bias=True, dropout_rate=0.5,
                 config=('conv', 'conv', 'max-pool', 'conv', 'conv', 'max-pool', 'conv', 'avg-pool', 'dense')):
        super(ConvLarge, self).__init__()
        
        layers, out_features = make_layers(config, input_shape, init_num_kernels, use_bn, hidden_bias, dropout_rate)
        
        self.input_layer = layers[0]
        self.hidden_layers = nn.Sequential(*layers[1:])
        
        self.output_layer = nn.Linear(out_features, num_classes, bias=output_bias)
        
    def forward(self, X):
        Z = self.input_layer(X)
        Z = self.hidden_layers(Z)
        return self.output_layer(Z)