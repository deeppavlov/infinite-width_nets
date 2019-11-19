import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain


def _test_gradient_normalization(optimizer):
    if isinstance(optimizer, (optim.SGD,)):
        return False
    elif isinstance(optimizer, (optim.Adadelta, optim.Adagrad, optim.Adam, optim.AdamW, optim.Adamax, optim.RMSprop)):
        return True
    else:
        raise ValueError("Unsupported optimizer: {}".format(type(optimizer)))
        
        
def _count_layers(layers):
    counter = 0
    for layer in layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            counter += 1
    return counter


def scale_hyperparams(input_layer, hidden_layers, output_layer, 
                      optimizer, width_factor, scaling_mode):
    
    #assert len(optimizer.param_groups) > 1, "there should be at least two param groups in optimizer; the first one should correspond to input layer"
    #assert optimizer.param_groups[0]['params'] == input_layer.parameters(), "the first param group of optimizer should correspond to input layer parameters"
    
    all_except_input_layers = chain(hidden_layers, [output_layer])
    #hidden_layer_count = _count_layers(hidden_layers)
    
    is_gradient_normalized = _test_gradient_normalization(optimizer)
    
    use_bn = False
    for layer in hidden_layers:
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            use_bn = True
            break
    
    if scaling_mode == 'default':
        weight_factor = 1
        lr_factor_input = 1
        lr_factor_hidden = 1
        lr_factor_output = 1
    
    elif scaling_mode == 'preserve_initial_logit_std':
        if hidden_layer_count > 0 or use_bn:
            raise NotImplementedError
        weight_factor = 1
        lr_factor_input = width_factor ** (0.5)
        lr_factor_hidden = width_factor ** (-0.5)
        lr_factor_output = width_factor ** (-0.5)
        
    elif scaling_mode == 'preserve_logit_mean':
        weight_factor = width_factor ** (-0.5)
        if is_gradient_normalized:
            lr_factor_input = 1
            lr_factor_hidden = width_factor ** (-1)
            lr_factor_output = width_factor ** (-1)
        else:
            if use_bn:
                lr_factor_input = 1
                lr_factor_hidden = width_factor ** (-2)
            else:
                lr_factor_input = width_factor
                lr_factor_hidden = 1
            lr_factor_output = width_factor ** (-1)
        
    elif scaling_mode == 'preserve_initial_logit_std_and_logit_mean':
        if hidden_layer_count > 0 or use_bn:
            raise NotImplementedError
        weight_factor = 1
        lr_factor_input = width_factor ** (0.5)
        lr_factor_hidden = width_factor ** (-0.5)
        lr_factor_output = width_factor ** (-0.5)
        
    else:
        raise ValueError("Unknown scaling mode: {}".format(scaling_mode))
        
    for layer in all_except_input_layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.weight.data = layer.weight.data * weight_factor
            if hasattr(layer, 'bias'):
                if layer.bias is not None:
                    raise NotImplementedError
                
    assert len(optimizer.param_groups) == 3
                
    optimizer.param_groups[0]['lr'] *= lr_factor_input
    optimizer.param_groups[1]['lr'] *= lr_factor_hidden
    optimizer.param_groups[2]['lr'] *= lr_factor_output
    