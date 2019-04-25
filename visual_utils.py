import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn


def detransform(img, mean: list, std: list):
    '''Detransform an img of a pytorch dataset.
    '''
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    # get the image back in [0,1] range (reverse Normalize(mean, std) process)
    denorm_img = img * std[...,None,None] + mean[...,None,None]
    # CxHxW -> HxWxC (reverse ToTensor() process)
    hwc_img = np.transpose(denorm_img, (1, 2, 0))
    # Tensor -> numpy.ndarray
    return hwc_img.numpy()

def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    '''
    ORIGINAL_SOURCE: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''
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

def _hook(layer: nn.Module):
    pass
