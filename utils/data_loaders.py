import numpy as np

import torch
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split

        
AVAILABLE_DATASETS = ['mnist', 'cifar10', 'cifar2']


class CIFAR2(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR2, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.data = self.data[self.targets < 2]
        self.targets = self.targets[self.targets < 2]
        print('dataset size = {}'.format(len(self.data)))


def _get_mean_and_std(dataset_name):
    if dataset_name == 'mnist':
        return 0.1307, 0.3081
    elif dataset_name == 'cifar10':
        return 0.4734, 0.2516
    elif dataset_name == 'cifar2':
        return 0.4734, 0.2516
    else:
        raise ValueError("unknown dataset: {}".format(dataset_name))
        
        
def _get_dataset_class(dataset_name):
    if dataset_name == 'mnist':
        return datasets.MNIST
    elif dataset_name == 'cifar10':
        return datasets.CIFAR10
    elif dataset_name == 'cifar2':
        return CIFAR2
    else:
        raise ValueError("unknown dataset: {}".format(dataset_name))
    

def get_shape(dataset_name):
    if dataset_name == 'mnist':
        return (1, 28, 28), 10
    elif dataset_name == 'cifar10':
        return (3, 32, 32), 10
    elif dataset_name == 'cifar2':
        return (3, 32, 32), 2
    else:
        raise ValueError("unknown dataset: {}".format(dataset_name))
        
        
def get_loaders(dataset_name, batch_size, train_size=None):
    dataset_mean, dataset_std = _get_mean_and_std(dataset_name)
    DatasetClass = _get_dataset_class(dataset_name)
    data_path = "data/{}".format(dataset_name)
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((dataset_mean,), (dataset_std,))])
    
    train_dataset = DatasetClass(
        data_path, train=True, download=True, transform=transform)
    test_dataset = DatasetClass(
        data_path, train=False, download=True, transform=transform)
    
    if train_size is None:
        sampler = None
        shuffle = True
    else:
        train_indices, _ = train_test_split(
            np.arange(len(train_dataset)), train_size=train_size, stratify=train_dataset.targets)
        sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        shuffle = False
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    test_loader_det = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_loader_det