import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
import numpy as np
from typing import Union, Callable, Optional, Dict
from tqdm import tqdm
from collections import namedtuple
import copy

__all__ = ['get_cls_to_idx', 'predict', 'train_one_epoch', 'validate', 'train', 'crc_dataset_and_loader',
           'train_data_obj', 'val_data_obj']


def get_cls_to_idx(dir: Union[str, Path]):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    cls_to_idx = {classes[i]: i for i in range(len(classes))}
    return cls_to_idx


def predict(model: nn.Module, inputs: Union[torch.Tensor, data.dataloader.DataLoader], device: torch.device):
    """If "inputs" is a dataloader, this function presumes that the dataset object has two return values,
       which means when we traverse the dataloader, each time we will get two values: input and target.
    """
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

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)

    tqdm_train_dl.write(f"lr: {lr} train loss: {avg_loss:.3f} train acc: {avg_acc:.3f} ")
    tqdm_train_dl.close()
    return {'train_loss': avg_loss, 'train_acc': avg_acc, 'lr': lr}


def validate(model: nn.Module, val_dl: data.dataloader.DataLoader, criterion: Callable,
             device: torch.device, need_confusion_matrix: bool = False) -> Dict[str, float]:
    model.eval()
    tqdm_val_dl = tqdm(val_dl)
    running_losses, running_corrects = 0.0, 0
    confusion_matrix = {'targets': [], 'preds': []}

    with torch.no_grad():
        for batch_nr, (inputs, targets) in enumerate(tqdm_val_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, targets)

            if need_confusion_matrix:
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
    return {'val_loss': val_loss, 'val_acc': val_acc, 'confusion_matrix': confusion_matrix}


def train(model: nn.Module, train_dl: data.dataloader.DataLoader, val_dl: data.dataloader.DataLoader,
          criterion: Callable, optimizer: optim.Optimizer, scheduler: Optional[object], epochs: int,
          device: torch.device):
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
            train_one_epoch(model, train_dl, criterion, optimizer, device)
            val_result = validate(model, val_dl, criterion, device)
            scheduler.step()

            if val_result['val_acc'] > best_acc:
                print(f'Better model saved at epoch {epoch}...')
                best_acc = val_result['val_acc']
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


def crc_dataset_and_loader(data_path, transforms, batch_size, shuffle, num_workers, drop_last):
    dataobj = namedtuple('dataobj', ['dataset', 'dataloader'])
    dataset = datasets.ImageFolder(root=data_path, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             drop_last=drop_last)
    return dataobj(dataset=dataset, dataloader=dataloader)


def train_data_obj(train_path, transforms, batch_size, shuffle=True, num_workers=0, drop_last=True):
    return crc_dataset_and_loader(train_path, transforms, batch_size, shuffle=shuffle, num_workers=num_workers,
                                  drop_last=drop_last)


def val_data_obj(val_path, transforms, batch_size, shuffle=False, num_workers=0, drop_last=False):
    return crc_dataset_and_loader(val_path, transforms, batch_size, shuffle=shuffle, num_workers=num_workers,
                                  drop_last=drop_last)
