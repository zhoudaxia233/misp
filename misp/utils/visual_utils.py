import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from typing import Dict, Tuple
from .utils import predict

__all__ = ['get_heatmap_tensor', 'detransform', 'plot_confusion_matrix', 'plot_stats']


def _get_activations(store: Dict):
    def hook(module, input, output):
        store['activations'] = output.detach()

    return hook


def _get_grads(store: Dict):
    def hook(module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            store['grads'] = grad_output[0].detach()
        elif isinstance(grad_output, torch.Tensor):
            store['grads'] = grad_output.detach()
        else:
            raise Exception("Something wrong with the grad_output.")

    return hook


def _hook(model: nn.Module, layer: nn.Module, img: torch.Tensor, category: int, device: torch.device):
    '''Get the activations and grads of the layer of the model.
    '''
    model.to(device)
    # register hooks
    store = {'activations': None, 'grads': None}
    forward_hook = layer.register_forward_hook(_get_activations(store))
    backward_hook = layer.register_backward_hook(_get_grads(store))

    try:
        # trigger them
        model.eval()
        one_batch_img = img[None, ...].to(device)
        pred = model(one_batch_img)
        pred[0, category].backward()

    finally:
        # remove hooks
        forward_hook.remove()
        backward_hook.remove()

    return store['activations'], store['grads']


def get_heatmap_tensor(model: nn.Module, layer: nn.Module, dataset: torch.utils.data.Dataset, idx: int,
                       device: torch.device, is_test: bool = False):
    if not is_test:
        acts, grads = _hook(model, layer, dataset[idx][0], dataset[idx][1], device)
    else:
        pred_cls = predict(model, dataset[idx][0], device)
        acts, grads = _hook(model, layer, dataset[idx][0], pred_cls, device)

    acts = acts.cpu()
    grads = grads.cpu()

    # simulate Global Average Pooling layer
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    # weight the channels by corresponding gradients (NxCxHxW, so dim 1 is the Channel dimension)
    for i in range(acts.size(dim=1)):
        acts[:, i, :, :] *= pooled_grads[i]

    # average the channels of the activations
    heatmap = torch.mean(acts, dim=1).squeeze()  # squeeze: the dimensions of input of size 1 are removed
    heatmap_relu = np.maximum(heatmap, 0)

    # normalize
    heatmap_relu /= torch.max(heatmap_relu)

    return heatmap_relu


def detransform(img, mean: Tuple = (0.485, 0.456, 0.406), std: Tuple = (0.229, 0.224, 0.225)):
    '''Detransform an img of a pytorch dataset.
    '''
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    # get the image back in [0,1] range (reverse Normalize(mean, std) process)
    denorm_img = img * std[..., None, None] + mean[..., None, None]
    # CxHxW -> HxWxC (reverse ToTensor() process)
    hwc_img = np.transpose(denorm_img, (1, 2, 0))
    # Tensor -> numpy.ndarray
    return hwc_img.numpy()


def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    """
    ORIGINAL_SOURCE: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_stats(stats, typ='loss', y_range=(0.0, 2.0)):
    epochs = len(list(stats.values())[0])
    x = range(epochs)
    if typ == 'loss':
        plt.plot(x, stats['train_loss'], label='train')
        plt.plot(x, stats['val_loss'], label='val')
        plt.ylabel('Loss')
        plt.ylim(y_range)
    elif typ == 'acc':
        plt.plot(x, stats['train_acc'], label='train')
        plt.plot(x, stats['val_acc'], label='val')
        plt.ylabel('Accuracy')
        plt.ylim(y_range)
    elif typ == 'lr':
        plt.plot(x, stats['lr'], label='lr')
        plt.ylabel('Learning Rate')
        plt.ylim(y_range)
    else:
        raise ValueError('Typ should be one of {loss, acc, lr}.')

    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
