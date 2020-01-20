from copy import copy, deepcopy

import numpy as np
import torch

from utils.scale_hyperparams import scale_hyperparams
from utils.perform_epoch import perform_epoch, get_logits
from utils.models import InitCorrectedModel, LinearizedModel, OptimizerForLinearizedModel


def get_model_with_modified_width(model_class, reference_model_kwargs, 
                                  width_arg_name='width', width_factor=1, init_corrected=False,
                                  device='cpu'):
    assert width_arg_name in reference_model_kwargs
    model_kwargs = deepcopy(reference_model_kwargs)
    model_kwargs[width_arg_name] = int(model_kwargs[width_arg_name] * width_factor)
    
    if init_corrected:
        model = InitCorrectedModel(model_class, model_kwargs).to(device)
    else:
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


def get_model_decomposition_term(model, model_init, output_inc, hidden_incs, input_inc):
    with torch.no_grad():
        model_decomposition_term = deepcopy(model_init)
        
        if input_inc:
            model_decomposition_term.input_layer.weight.data = model.input_layer.weight.data - model_init.input_layer.weight.data
            
        if output_inc:
            model_decomposition_term.output_layer.weight.data = model.output_layer.weight.data - model_init.output_layer.weight.data
        
        if hidden_incs:
            for layer_decomp, layer, layer_init in zip(model_decomposition_term.hidden_layers, model.hidden_layers, model_init.hidden_layers):
                if hasattr(layer, 'weight'):
                    layer_decomp.weight.data = layer.weight.data - layer_init.weight.data
                    
        return model_decomposition_term


def train_and_eval(model, model_init, optimizer, scaling_mode, train_loader, test_loader, test_loader_det, 
                   num_epochs, correction_epoch,
                   eval_every=None, test_batch_count=None, logits_batch_count=None, 
                   width_factor=1, device='cpu', print_progress=False, binary=False):
    train_losses = {}
    test_losses = {}
    
    train_accs = {}
    test_accs = {}
    
    logits = {}
    
    init_corrected = scaling_mode.endswith('init_corrected')
    
    loss_name = 'bce' if binary else 'ce'
    
    if model is None:
        model = model_init

        for epoch in range(num_epochs):
            if scaling_mode == 'linearized':
                if epoch == 0:
                    scale_hyperparams(
                        model.input_layer, model.hidden_layers, model.output_layer, 
                        optimizer=optimizer, width_factor=width_factor, scaling_mode=scaling_mode,
                        epoch=epoch, correction_epoch=correction_epoch
                    )
                    model = LinearizedModel(model)
                    optimizer = OptimizerForLinearizedModel(optimizer, model)
            else:
                scale_hyperparams(
                    model.input_layer, model.hidden_layers, model.output_layer, 
                    optimizer=optimizer, width_factor=width_factor, scaling_mode=scaling_mode,
                    epoch=epoch, correction_epoch=correction_epoch
                )
                if init_corrected:
                    scale_hyperparams(
                        model.input_layer_init, model.hidden_layers_init, model.output_layer_init, 
                        optimizer=None, width_factor=width_factor, scaling_mode=scaling_mode,
                        epoch=epoch, correction_epoch=correction_epoch
                    )

            if epoch == 0:
                model_init = deepcopy(model)

            model.train()
            train_loss, train_acc = perform_epoch(model, train_loader, optimizer=optimizer, device=device, loss_name=loss_name)
            train_losses[epoch] = train_loss
            train_accs[epoch] = train_acc

            if print_progress:
                print('Epoch {};'.format(epoch+1))
                print('train_loss = {:.4f}; train_acc = {:.2f}'.format(train_loss, train_acc*100))

            if eval_every is not None and (epoch+1) % eval_every == 0:
                model.eval()
                test_loss, test_acc = perform_epoch(model, test_loader, device=device, max_batch_count=test_batch_count, loss_name=loss_name)
                test_losses[epoch] = test_loss
                test_accs[epoch] = test_acc
                logits[epoch] = get_logits(model, test_loader_det, max_batch_count=logits_batch_count, device=device, loss_name=loss_name)
                if print_progress:
                    print('test_loss = {:.4f}; test_acc = {:.2f}'.format(test_loss, test_acc*100))
    
    elif model_init is None:
        raise ValueError("if model is not None, model_init should be not None too")

    model.eval()
    final_train_loss, final_train_acc = perform_epoch(model, train_loader, device=device, loss_name=loss_name)
    final_test_loss, final_test_acc = perform_epoch(model, test_loader, device=device, loss_name=loss_name)
    
    final_logits = get_logits(model, test_loader_det, device=device)

    init_train_loss, init_train_acc = perform_epoch(model_init, train_loader, device=device, loss_name=loss_name)
    init_test_loss, init_test_acc = perform_epoch(model_init, test_loader, device=device, loss_name=loss_name)
    
    init_logits = get_logits(model_init, test_loader_det, device=device)

    def get_suffix(output_inc, hidden_incs, input_inc):
        suffix = ['_']
        if output_inc:
            suffix += ['a']
        if hidden_incs:
            suffix += ['v']
        if input_inc:
            suffix += ['w']
        return ''.join(suffix)
    
    results_decomposition = {}
    for output_inc in [False, True]:
        for hidden_incs in [False, True]:
            for input_inc in [False, True]:
                suffix = get_suffix(output_inc, hidden_incs, input_inc)
                model_decomposition_term = get_model_decomposition_term(
                    model, model_init, output_inc, hidden_incs, input_inc
                )
                
                final_train_loss_term, final_train_acc_term = perform_epoch(
                    lambda x: model_decomposition_term(x, model_to_get_masks_from=model), train_loader, 
                    device=device, loss_name=loss_name
                )
                results_decomposition['final_train_loss'+suffix] = final_train_loss_term
                results_decomposition['final_train_acc'+suffix] = final_train_acc_term
    
                final_test_loss_term, final_test_acc_term = perform_epoch(
                    lambda x: model_decomposition_term(x, model_to_get_masks_from=model), test_loader, 
                    device=device, loss_name=loss_name
                )
                results_decomposition['final_test_loss'+suffix] = final_test_loss_term
                results_decomposition['final_test_acc'+suffix] = final_test_acc_term
    
                final_logits_term = get_logits(
                    lambda x: model_decomposition_term(x, model_to_get_masks_from=model),
                    test_loader_det, device=device
                )
                final_var_f_term = np.var(final_logits_term, axis=0)[0]
                results_decomposition['final_logits'+suffix] = final_logits_term
                results_decomposition['final_var_f'+suffix] = final_var_f_term
                
    with torch.no_grad():
        input_weight_mean_abs_inc = torch.mean(
            torch.norm(model.input_layer.weight.data - model_init.input_layer.weight.data, dim=1)
        ).cpu().item()
        hidden_weight_mean_abs_inc = []
        for layer, layer_init in zip(model.hidden_layers, model_init.hidden_layers):
            if hasattr(layer, 'weight'):
                hidden_weight_mean_abs_inc.append(
                    torch.mean(torch.abs(layer.weight.data - layer_init.weight.data)).cpu().item()
                )
        output_weight_mean_abs_inc = torch.mean(
            torch.abs(model.output_layer.weight.data - model_init.output_layer.weight.data)
        ).cpu().item()
    
    results = {
        'model_state_dict': model.cpu().state_dict(), 'model_init_state_dict': model_init.cpu().state_dict(),
        
        'train_losses': train_losses, 'train_accs': train_accs,
        'test_losses': test_losses, 'test_accs': test_accs,
        'logits': logits,
        
        'final_train_loss': final_train_loss, 'final_train_acc': final_train_acc,
        'final_test_loss': final_test_loss, 'final_test_acc': final_test_acc,
        'final_logits': final_logits,
        
        'init_train_loss': init_train_loss, 'init_train_acc': init_train_acc,
        'init_test_loss': init_test_loss, 'init_test_acc': init_test_acc,
        'init_logits': init_logits,
        
        'input_weight_mean_abs_inc': input_weight_mean_abs_inc,
        'hidden_weight_mean_abs_inc': hidden_weight_mean_abs_inc,
        'output_weight_mean_abs_inc': output_weight_mean_abs_inc
    }
    
    for key in results_decomposition.keys():
        results[key] = results_decomposition[key]
    
    return results