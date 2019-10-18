import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def perform_epoch(model, loader, optimizer=None, max_batch_count=None, device='cpu'):
    
    cum_loss = 0
    cum_acc = 0
    cum_batch_size = 0
    batch_count = 0
    
    with torch.no_grad() if optimizer is None else torch.enable_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            batch_size = X.shape[0]

            logits = model(X)
            loss = F.cross_entropy(logits, y)
            acc = torch.mean((torch.max(logits, dim=-1)[1] == y).float())
                
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            cum_loss += loss.item() * batch_size
            cum_acc += acc.item() * batch_size
            cum_batch_size += batch_size
            batch_count += 1
            
            if max_batch_count is not None and batch_count >= max_batch_count:
                break

    mean_loss = cum_loss / cum_batch_size
    mean_acc = cum_acc / cum_batch_size

    return mean_loss, mean_acc