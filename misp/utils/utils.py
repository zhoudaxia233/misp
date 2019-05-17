import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from typing import Union, Callable, Optional, Dict
from tqdm import tqdm
import copy

__all__ = ['copy_dir_tree', 'get_cls_to_idx', 'predict', 'train_one_epoch', 'validate', 'train']

def copy_dir_tree(src: str, dst: str, ignore_files: bool=False, symlinks: bool=False):
    src = Path(src)
    dst = Path(dst)

    print('Start copying directory tree ...')

    if os.path.isdir(dst):
        shutil.rmtree(dst)

    if ignore_files:
        def ignore_func(folder, folder_contents):
            return [f for f in folder_contents if not os.path.isdir(os.path.join(folder, f))]
        shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore_func)

    else:
        shutil.copytree(src, dst, symlinks=symlinks)

    print('Copying directory tree is done.')

def get_cls_to_idx(dir: Union[str, Path]):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    cls_to_idx = {classes[i]: i for i in range(len(classes))}
    return cls_to_idx

def predict(model: nn.Module, inputs: Union[torch.Tensor, data.dataloader.DataLoader], device: torch.device):
    '''If "inputs" is a dataloader, this function presumes that the dataset object has two return values,
       which means when we traverse the dataloader, each time we will get two values: input and target.
    '''
    model.eval()
    
    preds = []
    if isinstance(inputs, data.dataloader.DataLoader):
        with torch.no_grad():
            for batch_input, _ in inputs:
                batch_input = batch_input.to(device)

                outputs = model(batch_input)
                preds.extend(outputs.argmax(dim=1).cpu().tolist())
    elif isinstance(inputs, torch.Tensor):
        with torch.no_grad():
            inputs = inputs[None, ...].to(device)
            outputs = model(inputs)
            preds.append(outputs.argmax(dim=1).cpu().item())
    else:
        raise Exception(f'Your input type: {type(inputs)} is not supported.')
    
    return preds

def train_one_epoch(model: nn.Module, train_dl: data.dataloader.DataLoader, criterion: Callable,
    optimizer: optim.Optimizer, device: torch.device):
    model.train()        
    tqdm_train_dl = tqdm(train_dl)
    losses, accs = [], []
    
    for batch_nr, (inputs, targets) in enumerate(tqdm_train_dl):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_acc = torch.sum(preds == targets.data).item() / inputs.size(0)
            
        # record metrics
        losses.append(loss.item())
        accs.append(train_acc)

        tqdm_train_dl.set_description(f'Batch {batch_nr}')
        tqdm_train_dl.set_postfix(train_loss=loss.item(), train_acc=train_acc)
    lr = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]
    if len(lr) == 1:
        lr = lr[0]
    tqdm_train_dl.write(f"lr: {lr} train loss: {np.mean(losses):.3f} train acc: {np.mean(accs):.3f} ")
    tqdm_train_dl.close()

def validate(model: nn.Module, val_dl: data.dataloader.DataLoader, criterion: Callable,
    device: torch.device, confusion_matrix: Optional[Dict]=None) -> Dict[str, float]:
    model.eval()
    tqdm_val_dl = tqdm(val_dl)
    running_losses, running_corrects = 0.0, 0
    need_cm = True if confusion_matrix else False
    
    with torch.no_grad():
        for batch_nr, (inputs, targets) in enumerate(tqdm_val_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, targets)
            
            # if need confusion matrix
            if need_cm:
                if batch_nr == 0:
                    assert isinstance(confusion_matrix, dict)
                    assert set(confusion_matrix.keys())==set(['targets', 'preds'])
                    assert list(confusion_matrix.values())==[[], []]

                confusion_matrix['targets'].extend(targets.tolist())
                confusion_matrix['preds'].extend(preds.tolist())
            
            # record metrics
            running_losses += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets.data).item()
            
            tqdm_val_dl.set_description(f'Batch {batch_nr}')
            tqdm_val_dl.set_postfix(train_loss=loss.item())  
    val_loss = running_losses / len(val_dl.dataset)
    val_acc = running_corrects / len(val_dl.dataset)
    tqdm_val_dl.write(f"val loss: {val_loss:.3f} val acc: {val_acc:.3f} ")
    tqdm_val_dl.close()
    return {'val_loss': val_loss, 'val_acc': val_acc}

def train(model: nn.Module, train_dl: data.dataloader.DataLoader, val_dl: data.dataloader.DataLoader,
    criterion: Callable, optimizer: optim.Optimizer, scheduler: Optional[object], epochs: int, device: torch.device):    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    if not scheduler:
        for epoch in range(epochs):
            print(f'Epoch {epoch}: ')
            train_one_epoch(model, train_dl, criterion, optimizer, device)     
            val_result = validate(model, val_dl, criterion, device)

            if val_result['val_acc'] > best_acc:
                print(f'Better model saved at epoch {epoch}...')
                best_acc = val_result['val_acc']
                best_model_wts = copy.deepcopy(model.state_dict())
    
    elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        for epoch in range(epochs):
            print(f'Epoch {epoch}: ')
            train_one_epoch(model, train_dl, criterion, optimizer, device)     
            val_result = validate(model, val_dl, criterion, device)
            scheduler.step(val_result['val_acc'])

            if val_result['val_acc'] > best_acc:
                print(f'Better model saved at epoch {epoch}...')
                best_acc = val_result['val_acc']
                best_model_wts = copy.deepcopy(model.state_dict())
            
    else:
        for epoch in range(epochs):
            print(f'Epoch {epoch}: ')
            scheduler.step()
            train_one_epoch(model, train_dl, criterion, optimizer, device)     
            val_result = validate(model, val_dl, criterion, device)

            if val_result['val_acc'] > best_acc:
                print(f'Better model saved at epoch {epoch}...')
                best_acc = val_result['val_acc']
                best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    return model
