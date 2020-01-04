import argparse
import os
from collections import defaultdict
from copy import copy, deepcopy
import pickle

from utils.models import FCNet
from utils.data_loaders import get_shape, get_loaders
from utils.train_and_eval import get_model_with_modified_width, get_optimizer, train_and_eval
from utils.dict_utils import dict_to_defaultdict, defaultdict_to_dict

import numpy as np

import torch
import torch.optim as optim


model_class = FCNet

scaling_modes = ['default', 'mean_field', 'ntk', 'ntk_var1']
ref_widths = [32, 512, 8192, 2**15]
correction_epochs = [0]
real_widths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    
def get_optimizer_class_and_default_lr(optimizer_name):
    if optimizer_name == 'sgd':
        optimizer_class = optim.SGD
        default_lr = 1e-1
    elif optimizer_name == 'sgd_momentum':
        optimizer_class = SGDMomentum
        default_lr = 1e-1
    elif optimizer_name == 'rmsprop':
        optimizer_class = optim.RMSprop
        default_lr = 1e-3
    elif optimizer_name == 'adam':
        optimizer_class = optim.Adam
        default_lr = 1e-3
    else:
        raise ValueError
        
    return optimizer_class, default_lr


def get_log_dir():
    log_dir = os.path.join(
        'results', 'ref_width_dependence', '{}_{}'.format(args.dataset, args.train_size), 
        'num_hidden={}_bias={}_normalization={}'.format(args.num_hidden, args.bias, args.normalization), '{}_lr={}'.format(args.optimizer, args.lr))
    return log_dir


def assure_dir_exists(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except FileNotFoundError:
            tail, _ = os.path.split(path)
            assure_dir_exists(tail)
            os.mkdir(path)


def main(args):
    optimizer_class, default_lr = get_optimizer_class_and_default_lr(args.optimizer)
    if args.lr is None:
        args.lr = default_lr
    lr = float(args.lr)
    
    log_dir = get_log_dir()
    assure_dir_exists(log_dir)
    
    results_all_path = os.path.join(log_dir, 'results_all.dat')
    results_all = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))))
    if os.path.exists(results_all_path):
        with open(results_all_path, 'rb') as f:
            results_all = dict_to_defaultdict(pickle.load(f), results_all)
    
    input_shape, num_classes = get_shape(args.dataset)
    
    train_loader, test_loader, _ = get_loaders(args.dataset, args.batch_size, args.train_size)
    
    reference_model_kwargs = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'width': None,
        'num_hidden': args.num_hidden,
        'bias': args.bias,
        'normalization': args.normalization,
    }
    
    for scaling_mode in scaling_modes:
        
        for ref_width in (ref_widths if scaling_mode != 'default' else [None]):
            reference_model_kwargs['width'] = ref_width
                
            for correction_epoch in (correction_epochs if scaling_mode.startswith('mean_field') else [None]):
                if correction_epoch == 0 and scaling_mode == 'mean_field_var1':
                    continue
            
                for real_width in real_widths:
                    if ref_width is None:
                        width_factor = 1
                        reference_model_kwargs['width'] = real_width
                    else:
                        width_factor = real_width / ref_width

                    if scaling_mode == 'ntk_var1':
                        corrected_num_epochs = int(args.num_epochs * width_factor ** (-0.5))
                    else:
                        corrected_num_epochs = args.num_epochs
                    print('corrected_num_epochs = {}'.format(corrected_num_epochs))
                        
                    for seed in range(args.num_seeds):
                        print('ref_width = {}'.format(ref_width))
                        print('scaling_mode = {}'.format(scaling_mode))
                        print('correction_epoch = {}'.format(correction_epoch))
                        print('real_width = {}'.format(real_width))
                        print('seed = {}'.format(seed))
                        
                        if results_all[scaling_mode][ref_width][correction_epoch][real_width][seed] is not None:
                            print('already done\n')
                            continue
                        
                        torch.manual_seed(seed+100)
                        np.random.seed(seed+100)

                        model = get_model_with_modified_width(
                            model_class, reference_model_kwargs, width_arg_name='width',
                            width_factor=width_factor, device=torch.device(args.device))

                        optimizer = get_optimizer(optimizer_class, {'lr': lr}, model)

                        results = train_and_eval(
                            model, optimizer, scaling_mode, train_loader, test_loader, 
                            corrected_num_epochs, correction_epoch, width_factor=width_factor, 
                            device=torch.device(args.device), print_progress=args.print_progress)
                        
                        print('final_train_loss = {:.4f}; final_train_acc = {:.2f}'.format(results['final_train_loss'], results['final_train_acc']*100))
                        print('final_test_loss = {:.4f}; final_test_acc = {:.2f}'.format(results['final_test_loss'], results['final_test_acc']*100))
                        print()
                        
                        results_all[scaling_mode][ref_width][correction_epoch][real_width][seed] = copy(results)
                        
                        with open(results_all_path, 'wb') as f:
                            pickle.dump(defaultdict_to_dict(results_all), f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cpu')
    argparser.add_argument('--dataset', type=str, default='mnist')
    argparser.add_argument('--train_size', type=int, default=1000)
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument('--num_epochs', type=int, default=50)
    argparser.add_argument('--num_seeds', type=int, default=5)
    argparser.add_argument('--num_hidden', type=int, default=1)
    argparser.add_argument('--bias', type=bool, default=False)
    argparser.add_argument('--normalization', type=str, default='none')
    argparser.add_argument('--optimizer', type=str, default='sgd')
    argparser.add_argument('--lr', default=None)
    argparser.add_argument('--print_progress', type=bool, default=False)

    args = argparser.parse_args()

    main(args)