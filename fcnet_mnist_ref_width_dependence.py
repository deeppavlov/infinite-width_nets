import argparse
import os
from collections import defaultdict
from copy import copy, deepcopy

from utils.models import FCNet
from utils.data_loaders import get_shape, get_loaders
from utils.train_and_eval import get_model_width_modified_width, get_optimizer, train_and_eval


model_class = FCNet

ref_widths = [32, 512, 8192]
scaling_modes = ['default', 'mean_field']
correction_epochs = [0]
real_widths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    
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
        'num_hidden={}_bias={}_normalization={}'.format(args.num_hidden, args.bias, args.normalization), args.optimizer)
    return log_dir


def assure_dir_exists(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except FileNotFoundError:
            tail, _ = os.path.split(path)
            assure_path_exists(tail)
            os.mkdir(path)


def main(args):
    log_dir = get_log_dir()
    assure_dir_exists(log_dir)
    
    results_all_path = os.path.join(log_dir, 'results_all.dat')
    if not os.path.exists(results_all_path):
        results_all = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))))
    else:
        with open(results_all_path, 'rb') as f:
            results_all = pickle.load(f)
    
    input_shape, num_classes = get_shape(dataset_name)
    
    train_loader, test_loader, _ = get_loaders(dataset_name, args.batch_size, args.train_size)
    
    reference_model_kwargs = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'width': None,
        'num_hidden': args.num_hidden,
        'bias': args.bias,
        'normalization': args.normalization,
    }
    
    optimizer_class, default_lr = get_optimizer_class_and_default_lr(args.optimizer)
    if args.lr is None:
        lr = default_lr
    else:
        lr = args.lr
    
    for scaling_mode in scaling_modes:
        
        for ref_width in (ref_widths if scaling_mode != 'default' else [None]):
            reference_model_kwargs['width'] = ref_width
                
            for correction_epoch in (correction_epochs if scaling_mode != 'default' else [None]):
            
                for real_width in real_widths:
                    width_factor = real_width / ref_width

                    for seed in range(args.num_seeds):
                        print('ref_width = {}'.format(ref_width))
                        print('scaling_mode = {}'.format(scaling_mode))
                        print('correction_epoch = {}'.format(correction_epoch))
                        print('real_width = {}'.format(real_width))
                        print('seed = {}'.format(seed))
                        
                        if results_all[scaling_mode][ref_width][correction_epoch][real_width][seed] is not None:
                            print('already done\n')
                            continue
                        
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        model = get_model_width_modified_width(
                            model_class, reference_model_kwargs, width_arg_name='width',
                            width_factor=width_factor, device=args.device)

                        optimizer = get_optimizer(optimizer_class, {'lr': lr}, model)

                        results = train_and_eval(
                            model, optimizer, scaling_mode, train_loader, test_loader, 
                            args.num_epochs, correction_epoch, width_factor=width_factor, 
                            device=args.device, print_progress=args.print_progress)
                        
                        print('final_train_loss = {:.4f}; final_train_acc = {:.2f}'.format(results['final_train_loss'], results['final_train_acc']*100))
                        print('final_test_loss = {:.4f}; final_test_acc = {:.2f}'.format(results['final_test_loss'], results['final_test_acc']*100))
                        print()
                        
                        results_all[scaling_mode][ref_width][correction_epoch][real_width][seed] = copy(results)
                        
                        with open(results_all_path, 'wb') as f:
                            pickle.dump(results_all, f)


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