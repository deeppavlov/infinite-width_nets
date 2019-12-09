from copy import copy, deepcopy

import numpy as np
import torch

from utils.scale_hyperparams import scale_hyperparams
from utils.perform_epoch import perform_epoch


def get_model_with_modified_width(model_class, reference_model_kwargs, width_arg_name='width', width_factor=1, device='cpu'):
    assert width_arg_name in reference_model_kwargs
    model_kwargs = deepcopy(reference_model_kwargs)
    model_kwargs[width_arg_name] = int(model_kwargs[width_arg_name] * width_factor)
    
    model = model_class(**model_kwargs).to(device)
    return model


def get_optimizer(optimizer_class, optimizer_kwargs, model):
    assert hasattr(model, 'input_layer')
    assert hasattr(model, 'hidden_layers')
    assert hasattr(model, 'output_layer')
    optimizer = optimizer_class(
        [
            {'params': model.input_layer.parameters()},
            {'params': model.hidden_layers.parameters()},
            {'params': model.output_layer.parameters()}
        ], **optimizer_kwargs
    )
    return optimizer


def train_and_eval(model, optimizer, scaling_mode, train_loader, test_loader, num_epochs, correction_epoch, 
                   eval_every=None, test_batch_count=None, width_factor=1, device='cpu', print_progress=False):
    train_losses = {}
    test_losses = {}
    
    train_accs = {}
    test_accs = {}

    for epoch in range(num_epochs):
        scale_hyperparams(
            model.input_layer, model.hidden_layers, model.output_layer, 
            optimizer=optimizer, width_factor=width_factor, scaling_mode=scaling_mode,
            epoch=epoch, correction_epoch=correction_epoch
        )
        
        model.train()
        train_loss, train_acc = perform_epoch(model, train_loader, optimizer=optimizer, device=device)
        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc
        
        if print_progress:
            print('Epoch {};'.format(epoch+1))
            print('train_loss = {:.4f}; train_acc = {:.2f}'.format(train_loss, train_acc*100))

        if eval_every is not None and (epoch+1) % eval_every == 0:
            model.eval()
            test_loss, test_acc = perform_epoch(model, test_loader, device=device, max_batch_count=test_batch_count)
            test_losses[epoch] = test_loss
            test_accs[epoch] = test_acc
            if print_progress:
                print('test_loss = {:.4f}; test_acc = {:.2f}'.format(test_loss, test_acc*100))

    model.eval()
    final_train_loss, final_train_acc = perform_epoch(model, train_loader, device=device)
    final_test_loss, final_test_acc = perform_epoch(model, test_loader, device=device)

    results = {
        'train_losses': train_losses, 'train_accs': train_accs,
        'test_losses': test_losses, 'test_accs': test_accs,
        'final_train_loss': final_train_loss, 'final_train_acc': final_train_acc,
        'final_test_loss': final_test_loss, 'final_test_acc': final_test_acc
    }
    return results