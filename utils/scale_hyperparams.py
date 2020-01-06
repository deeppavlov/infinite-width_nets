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


def _normalize_scaling_mode(scaling_mode):
    scaling_mode = scaling_mode.split('_init_corrected')[0]
    components = scaling_mode.split('_q=')
    scaling_mode = components[0]
    if len(components) > 1:
        q = float(components[1])
    else:
        q = None
    return scaling_mode, q


def scale_hyperparams(input_layer, hidden_layers, output_layer, 
                      optimizer, width_factor, scaling_mode, epoch, correction_epoch=0):
    
    scaling_mode, q = _normalize_scaling_mode(scaling_mode)
    #print(scaling_mode, q)
    
    all_except_input_layers = chain(hidden_layers, [output_layer])
    #hidden_layer_count = _count_layers(hidden_layers)
    
    if optimizer is not None:
        is_gradient_normalized = _test_gradient_normalization(optimizer)
    
#     use_bn = False
#     for layer in hidden_layers:
#         if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             use_bn = True
#             break
    
    weight_factor = 1
    lr_factor_input = 1
    lr_factor_hidden = 1
    lr_factor_output = 1

    if scaling_mode == 'default':
        pass
    
    elif scaling_mode == 'mean_field':
        if epoch == 0:
            weight_factor *= 1
            if optimizer is not None:
                if is_gradient_normalized:
                    lr_factor_input *= 1
                    lr_factor_hidden *= width_factor ** (-0.5)
                    lr_factor_output *= width_factor ** (-0.5)
                else:
                    lr_factor_input *= width_factor ** 0.5
                    lr_factor_hidden *= 1
                    lr_factor_output *= width_factor ** (-0.5)
        
        if correction_epoch is not None and epoch == correction_epoch:
            weight_factor *= width_factor ** (-0.5)
            if optimizer is not None:
                if is_gradient_normalized:
                    lr_factor_input *= 1
                    lr_factor_hidden *= width_factor ** (-0.5)
                    lr_factor_output *= width_factor ** (-0.5)
                else:
                    lr_factor_input *= width_factor ** 0.5
                    lr_factor_hidden *= 1
                    lr_factor_output *= width_factor ** (-0.5)
        
    elif scaling_mode == 'mean_field_var1':
        if epoch == 0:
            weight_factor *= width_factor ** (-0.5)
            if optimizer is not None:
                if is_gradient_normalized:
                    raise NotImplementedError
                else:
                    lr_factor_input *= width_factor ** 1.5
                    lr_factor_hidden *= 1
                    lr_factor_output *= width_factor ** (-0.5)
        
        if correction_epoch is not None and epoch == correction_epoch:
            weight_factor *= 1
            if optimizer is not None:
                if is_gradient_normalized:
                    raise NotImplementedError
                else:
                    lr_factor_input *= width_factor ** (-0.5)
                    lr_factor_hidden *= 1
                    lr_factor_output *= width_factor ** (-0.5)
        
    elif scaling_mode in {'ntk', 'linearized'}:
        if epoch == 0:
            weight_factor = 1
            if optimizer is not None:
                if is_gradient_normalized:
                    lr_factor_input = width_factor ** (-0.5)
                    lr_factor_hidden = width_factor ** (-1)
                    lr_factor_output = width_factor ** (-1)
                else:
                    lr_factor_input = 1
                    lr_factor_hidden = width_factor ** (-1)
                    lr_factor_output = width_factor ** (-1)
        
    elif scaling_mode == 'intermediate':
        if epoch == 0:
            weight_factor = width_factor ** (0.5-q)
            if optimizer is not None:
                if is_gradient_normalized:
                    lr_factor_input = width_factor ** (q-1)
                    lr_factor_hidden = width_factor ** (-1)
                    lr_factor_output = width_factor ** (-1)
                else:
                    lr_factor_input = width_factor ** (2*q-1)
                    lr_factor_hidden = width_factor ** (2*q-2)
                    lr_factor_output = width_factor ** (-1)
        
    else:
        raise ValueError("Unknown scaling mode: {}".format(scaling_mode))
        
    for layer in all_except_input_layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.weight.data = layer.weight.data * weight_factor
            if hasattr(layer, 'bias'):
                if layer.bias is not None:
                    raise NotImplementedError
                
    if optimizer is not None:
        assert len(optimizer.param_groups) == 3

        optimizer.param_groups[0]['lr'] *= lr_factor_input
        optimizer.param_groups[1]['lr'] *= lr_factor_hidden
        optimizer.param_groups[2]['lr'] *= lr_factor_output
    