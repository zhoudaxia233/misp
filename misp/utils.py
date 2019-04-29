import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
from typing import Union


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
