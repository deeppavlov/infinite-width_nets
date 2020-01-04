import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitCorrectedModel(nn.Module):
    def __init__(self, model_class, model_kwargs):
        super(InitCorrectedModel, self).__init__()
        
        self.model_scaled = model_class(**model_kwargs)
        self.model_init = model_class(**model_kwargs)
        self.model_init_scaled = model_class(**model_kwargs)
        
        self.input_layer = self.model_scaled.input_layer
        self.hidden_layers = self.model_scaled.hidden_layers
        self.output_layer = self.model_scaled.output_layer
        
        self.input_layer_init = self.model_init_scaled.input_layer
        self.hidden_layers_init = self.model_init_scaled.hidden_layers
        self.output_layer_init = self.model_init_scaled.output_layer
        
    def forward(self, X):
        return self.model_scaled(X) - self.model_init_scaled(X) + self.model_init(X)


def get_normalization_layer(width, normalization, dim):
    if normalization == 'none':
        layer = nn.Identity()
        
    elif normalization.startswith('batch'):
        if normalization == 'batch':
            affine = False
        elif normalization == 'batch_affine':
            affine = True
        else:
            raise ValueError
            
        if dim == 1:
            layer = nn.BatchNorm1d(width, affine=affine)
        elif dim == 2:
            layer = nn.BatchNorm2d(width, affine=affine)
        elif dim == 3:
            layer = nn.BatchNorm3d(width, affine=affine)
        else:
            raise ValueError
        
    elif normalization.startswith('layer'):
        if normalization == 'layer':
            affine = False
        elif normalization == 'layer_affine':
            affine = True
        else:
            raise ValueError
            
        layer = nn.LayerNorm(width, elementwise_affine=affine)
        
    else:
        raise ValueError
        
    return layer


def get_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        return nn.ReLU
    elif nonlinearity == 'lrelu':
        return nn.LeakyReLU
    elif nonlinearity == 'softplus':
        return nn.Softplus
    elif isinstance(nonlinearity, nn.Module):
        return nonlinearity
    else:
        raise ValueError


class FCNet(nn.Module):
    def __init__(self, input_shape, num_classes, width, 
                 num_hidden=1, bias=True, normalization='none', nonlinearity='relu'):
        super(FCNet, self).__init__()
        
        nonlinearity = get_nonlinearity(nonlinearity)
        
        input_size = int(np.prod(input_shape))
        
        self.input_layer = nn.Linear(input_size, width, bias=bias)
        
        hidden_layers = []
        hidden_layers.append(nonlinearity)
        hidden_layers.append(get_normalization_layer(width, normalization, dim=1))
        
        for _ in range(num_hidden-1):
            hidden_layers.append(nn.Linear(width, width, bias=bias))
            hidden_layers.append(nonlinearity)
            hidden_layers.append(get_normalization_layer(width, normalization, dim=1))
            
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.output_layer = nn.Linear(width, num_classes, bias=bias)
        
    def forward(self, X):
        Z = self.input_layer(X.view(X.shape[0], -1))
        Z = self.hidden_layers(Z)
        return self.output_layer(Z.view(Z.shape[0], -1))
    
    
def make_layers(config, input_shape, init_num_kernels, use_bn, bias, dropout_rate, bn_affine):
    out_channels, h, w = input_shape
    num_kernels = init_num_kernels
    layers = []
    for layer_name in config:
        if layer_name == 'conv':
            in_channels, out_channels = out_channels, num_kernels
            layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias), nn.LeakyReLU(0.1)]
            if use_bn:
                layers += [nn.BatchNorm2d(out_channels, affine=bn_affine)]
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
                layers += [nn.BatchNorm1d(out_channels, affine=bn_affine)]
                
    return layers, out_channels

    
class ConvLarge(nn.Module):
    def __init__(self, input_shape, num_classes, init_num_kernels, use_bn=True, hidden_bias=False, output_bias=True, dropout_rate=0.5, bn_affine=True,
                 config=('conv', 'conv', 'max-pool', 'conv', 'conv', 'max-pool', 'conv', 'avg-pool', 'dense')):
        super(ConvLarge, self).__init__()
        
        layers, out_features = make_layers(config, input_shape, init_num_kernels, use_bn, hidden_bias, dropout_rate, bn_affine)
        
        self.input_layer = layers[0]
        self.hidden_layers = nn.Sequential(*layers[1:])
        
        self.output_layer = nn.Linear(out_features, num_classes, bias=output_bias)
        
    def forward(self, X):
        Z = self.input_layer(X)
        Z = self.hidden_layers(Z)
        return self.output_layer(Z)